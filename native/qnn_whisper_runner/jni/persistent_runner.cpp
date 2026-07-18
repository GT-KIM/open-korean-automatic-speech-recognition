#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// QnnSampleApp intentionally keeps graph handles private. The benchmark uses the
// official loader/lifecycle implementation but needs direct tensor access for
// Whisper's autoregressive cache loop.
#define private public
#include "QnnSampleApp.hpp"
#undef private

#include "DynamicLoadUtil.hpp"
#include "IOTensor.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QnnTypeMacros.hpp"

namespace {

using App = qnn::tools::sample_app::QnnSampleApp;
using Status = qnn::tools::sample_app::StatusCode;
using Clock = std::chrono::steady_clock;

constexpr int32_t kDecoderStartToken = 50258;
constexpr int32_t kEndToken = 50257;
constexpr int32_t kForcedTokens[] = {50264, 50359, 50363};  // Korean, transcribe, no timestamps
constexpr uint8_t kSelfCacheZeroPoint = 128;
constexpr uint16_t kMaskVisible = 65535;
constexpr int kDecoderSteps = 199;

void fail(const std::string& message) {
  std::cerr << "ERROR " << message << std::endl;
  std::exit(EXIT_FAILURE);
}

double milliseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

Qnn_ClientBuffer_t buffer(Qnn_Tensor_t& tensor) {
  return QNN_TENSOR_GET_CLIENT_BUF(tensor);
}

std::unordered_map<std::string, Qnn_Tensor_t*> indexTensors(Qnn_Tensor_t* tensors,
                                                            uint32_t count) {
  std::unordered_map<std::string, Qnn_Tensor_t*> result;
  for (uint32_t i = 0; i < count; ++i) {
    const char* name = QNN_TENSOR_GET_NAME(tensors[i]);
    if (name != nullptr) {
      result.emplace(name, &tensors[i]);
    }
  }
  return result;
}

Qnn_Tensor_t& requireTensor(std::unordered_map<std::string, Qnn_Tensor_t*>& tensors,
                            const std::string& name) {
  auto found = tensors.find(name);
  if (found == tensors.end()) {
    fail("missing tensor: " + name);
  }
  return *found->second;
}

class LoadedGraph {
 public:
  LoadedGraph(const qnn::tools::sample_app::QnnFunctionPointers& functions,
              void* backendLibraryHandle,
              const std::string& contextPath)
      : app_(functions,
             "",
             "",
             backendLibraryHandle,
             "",
             false,
             qnn::tools::iotensor::OutputDataType::NATIVE_ONLY,
             qnn::tools::iotensor::InputDataType::NATIVE,
             qnn::tools::sample_app::ProfilingLevel::OFF,
             false,
             contextPath) {
    if (app_.initializeBackend() != Status::SUCCESS) fail("backend initialization failed");
    const auto property = app_.isDevicePropertySupported();
    if (property != Status::FAILURE && app_.createDevice() != Status::SUCCESS) {
      fail("device creation failed");
    }
    deviceCreated_ = property != Status::FAILURE;
    if (app_.createFromBinary() != Status::SUCCESS) fail("context loading failed: " + contextPath);
    if (app_.m_graphsCount != 1) fail("expected exactly one graph in " + contextPath);

    graph_ = (*app_.m_graphsInfo)[0];
    if (io_.setupInputAndOutputTensors(&inputs_, &outputs_, graph_) !=
        qnn::tools::iotensor::StatusCode::SUCCESS) {
      fail("tensor allocation failed: " + contextPath);
    }
    inputByName_ = indexTensors(inputs_, graph_.numInputTensors);
    outputByName_ = indexTensors(outputs_, graph_.numOutputTensors);
  }

  ~LoadedGraph() {
    io_.tearDownInputAndOutputTensors(
        inputs_, outputs_, graph_.numInputTensors, graph_.numOutputTensors);
    app_.freeContext();
    if (deviceCreated_) app_.freeDevice();
    app_.terminateBackend();
  }

  void execute() {
    const auto status = app_.m_qnnFunctionPointers.qnnInterface.graphExecute(graph_.graph,
                                                                             inputs_,
                                                                             graph_.numInputTensors,
                                                                             outputs_,
                                                                             graph_.numOutputTensors,
                                                                             nullptr,
                                                                             nullptr);
    if (status != QNN_GRAPH_NO_ERROR) fail("graph execution failed: " + std::to_string(status));
  }

  Qnn_Tensor_t& input(const std::string& name) { return requireTensor(inputByName_, name); }
  Qnn_Tensor_t& output(const std::string& name) { return requireTensor(outputByName_, name); }

 private:
  App app_;
  bool deviceCreated_ = false;
  qnn_wrapper_api::GraphInfo_t graph_{};
  qnn::tools::iotensor::IOTensor io_;
  Qnn_Tensor_t* inputs_ = nullptr;
  Qnn_Tensor_t* outputs_ = nullptr;
  std::unordered_map<std::string, Qnn_Tensor_t*> inputByName_;
  std::unordered_map<std::string, Qnn_Tensor_t*> outputByName_;
};

struct Options {
  std::string backend;
  std::string system;
  std::string encoder;
  std::string decoder;
  std::string features;
  int start = 1;
  int count = 0;
};

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key = argv[i];
    const std::string value = argv[i + 1];
    if (key == "--backend") options.backend = value;
    else if (key == "--system") options.system = value;
    else if (key == "--encoder") options.encoder = value;
    else if (key == "--decoder") options.decoder = value;
    else if (key == "--features") options.features = value;
    else if (key == "--start") options.start = std::stoi(value);
    else if (key == "--count") options.count = std::stoi(value);
    else fail("unknown argument: " + key);
  }
  if (options.backend.empty() || options.system.empty() || options.encoder.empty() ||
      options.decoder.empty() || options.features.empty() || options.start <= 0 ||
      options.count < options.start) {
    fail("required: --backend --system --encoder --decoder --features --count");
  }
  return options;
}

void copyTensor(Qnn_Tensor_t& destination, Qnn_Tensor_t& source, const std::string& name) {
  const auto dst = buffer(destination);
  const auto src = buffer(source);
  if (dst.dataSize != src.dataSize) fail("tensor size mismatch: " + name);
  std::memcpy(dst.data, src.data, dst.dataSize);
}

std::vector<int32_t> decode(LoadedGraph& decoder, LoadedGraph& encoder, double& decoderMs) {
  for (int layer = 0; layer < 12; ++layer) {
    for (const char prefix : {'k', 'v'}) {
      const std::string cross = std::string(1, prefix) + "_cache_cross_" + std::to_string(layer);
      copyTensor(decoder.input(cross), encoder.output(cross), cross);
      const std::string self = std::string(1, prefix) + "_cache_self_" +
                               std::to_string(layer) + "_in";
      const auto cache = buffer(decoder.input(self));
      std::memset(cache.data, kSelfCacheZeroPoint, cache.dataSize);
    }
  }

  auto tokenBuffer = buffer(decoder.input("input_ids"));
  auto positionBuffer = buffer(decoder.input("position_ids"));
  auto maskBuffer = buffer(decoder.input("attention_mask"));
  std::memset(maskBuffer.data, 0, maskBuffer.dataSize);
  auto* token = static_cast<int32_t*>(tokenBuffer.data);
  auto* position = static_cast<int32_t*>(positionBuffer.data);
  auto* mask = static_cast<uint16_t*>(maskBuffer.data);
  const size_t maskElements = maskBuffer.dataSize / sizeof(uint16_t);
  if (tokenBuffer.dataSize < sizeof(int32_t) || positionBuffer.dataSize < sizeof(int32_t) ||
      maskElements < static_cast<size_t>(kDecoderSteps + 1)) {
    fail("unexpected decoder scalar/mask tensor size");
  }

  std::vector<int32_t> tokens{kDecoderStartToken};
  decoderMs = 0.0;
  for (int step = 0; step < kDecoderSteps; ++step) {
    *token = tokens.back();
    *position = step;
    mask[kDecoderSteps - step] = kMaskVisible;

    const auto started = Clock::now();
    decoder.execute();
    decoderMs += milliseconds(started, Clock::now());

    int32_t next = 0;
    if (step < static_cast<int>(sizeof(kForcedTokens) / sizeof(kForcedTokens[0]))) {
      next = kForcedTokens[step];
    } else {
      const auto logitsBuffer = buffer(decoder.output("logits"));
      const auto* logits = static_cast<const uint16_t*>(logitsBuffer.data);
      const size_t count = logitsBuffer.dataSize / sizeof(uint16_t);
      next = static_cast<int32_t>(std::max_element(logits, logits + count) - logits);
    }
    tokens.push_back(next);
    if (next == kEndToken) break;

    for (int layer = 0; layer < 12; ++layer) {
      for (const char prefix : {'k', 'v'}) {
        const std::string base = std::string(1, prefix) + "_cache_self_" +
                                 std::to_string(layer);
        copyTensor(decoder.input(base + "_in"), decoder.output(base + "_out"), base);
      }
    }
  }
  return tokens;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parseOptions(argc, argv);
  qnn::tools::sample_app::QnnFunctionPointers functions{};
  void* backendLibraryHandle = nullptr;
  void* modelLibraryHandle = nullptr;
  if (qnn::tools::dynamicloadutil::getQnnFunctionPointers(options.backend,
                                                          "",
                                                          &functions,
                                                          &backendLibraryHandle,
                                                          false,
                                                          &modelLibraryHandle) !=
      qnn::tools::dynamicloadutil::StatusCode::SUCCESS) {
    fail("could not load QNN backend");
  }
  if (qnn::tools::dynamicloadutil::getQnnSystemFunctionPointers(options.system, &functions) !=
      qnn::tools::dynamicloadutil::StatusCode::SUCCESS) {
    fail("could not load QNN system library");
  }

  FILE* featureFile = std::fopen(options.features.c_str(), "rb");
  if (featureFile == nullptr) fail("could not open features file");

  {
    LoadedGraph encoder(functions, backendLibraryHandle, options.encoder);
    LoadedGraph decoder(functions, backendLibraryHandle, options.decoder);
    const auto featureBuffer = buffer(encoder.input("input_features"));
    const auto offset = static_cast<long>(options.start - 1) *
                        static_cast<long>(featureBuffer.dataSize);
    if (std::fseek(featureFile, offset, SEEK_SET) != 0) fail("could not seek features file");

    for (int index = options.start; index <= options.count; ++index) {
      if (std::fread(featureBuffer.data, 1, featureBuffer.dataSize, featureFile) !=
          featureBuffer.dataSize) {
        fail("short feature read at sample " + std::to_string(index));
      }
      const auto encoderStarted = Clock::now();
      encoder.execute();
      const double encoderMs = milliseconds(encoderStarted, Clock::now());
      double decoderMs = 0.0;
      const std::vector<int32_t> tokens = decode(decoder, encoder, decoderMs);

      std::cout << "RESULT {\"index\":" << index << ",\"tokens\":[";
      for (size_t i = 0; i < tokens.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << tokens[i];
      }
      std::cout << "],\"encoder_ms\":" << encoderMs << ",\"decoder_ms\":" << decoderMs
                << "}" << std::endl;
      if (index % 25 == 0 || index == options.count) {
        std::cerr << "PROGRESS " << index << '/' << options.count << std::endl;
      }
    }
  }
  std::fclose(featureFile);
  if (backendLibraryHandle != nullptr) pal::dynamicloading::dlClose(backendLibraryHandle);
  if (modelLibraryHandle != nullptr) pal::dynamicloading::dlClose(modelLibraryHandle);
  return EXIT_SUCCESS;
}
