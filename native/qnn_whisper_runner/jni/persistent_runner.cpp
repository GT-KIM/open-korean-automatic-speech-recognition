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

constexpr uint8_t kSelfCacheZeroPoint = 128;
constexpr uint16_t kMaskVisible = 65535;
constexpr int kDecoderSteps = 199;
constexpr uint16_t kFloat16NegativeHundred = 0xD640;

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
    if (!setupTensors(&inputs_, graph_.numInputTensors, graph_.inputTensors) ||
        !setupTensors(&outputs_, graph_.numOutputTensors, graph_.outputTensors)) {
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
  bool hasInput(const std::string& name) const { return inputByName_.count(name) != 0; }

 private:
  static size_t elementSize(Qnn_DataType_t dataType) {
    switch (dataType) {
      case QNN_DATATYPE_BOOL_8:
      case QNN_DATATYPE_INT_8:
      case QNN_DATATYPE_UINT_8:
      case QNN_DATATYPE_SFIXED_POINT_8:
      case QNN_DATATYPE_UFIXED_POINT_8:
        return 1;
      case QNN_DATATYPE_FLOAT_16:
      case QNN_DATATYPE_INT_16:
      case QNN_DATATYPE_UINT_16:
      case QNN_DATATYPE_SFIXED_POINT_16:
      case QNN_DATATYPE_UFIXED_POINT_16:
        return 2;
      case QNN_DATATYPE_FLOAT_32:
      case QNN_DATATYPE_INT_32:
      case QNN_DATATYPE_UINT_32:
      case QNN_DATATYPE_SFIXED_POINT_32:
      case QNN_DATATYPE_UFIXED_POINT_32:
        return 4;
      case QNN_DATATYPE_INT_64:
      case QNN_DATATYPE_UINT_64:
        return 8;
      default:
        fail("unsupported tensor data type: " + std::to_string(dataType));
        return 0;
    }
  }

  static bool setupTensors(Qnn_Tensor_t** tensors,
                           uint32_t count,
                           Qnn_Tensor_t* wrappers) {
    if (count == 0) return true;
    *tensors = static_cast<Qnn_Tensor_t*>(std::calloc(count, sizeof(Qnn_Tensor_t)));
    if (*tensors == nullptr) return false;
    for (uint32_t index = 0; index < count; ++index) {
      Qnn_Tensor_t& tensor = (*tensors)[index];
      tensor = QNN_TENSOR_INIT;
      if (!qnn::tools::sample_app::deepCopyQnnTensorInfo(&tensor, &wrappers[index])) {
        return false;
      }
      QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_RAW);
      size_t elements = 1;
      const auto* dimensions = QNN_TENSOR_GET_DIMENSIONS(tensor);
      for (uint32_t dimension = 0; dimension < QNN_TENSOR_GET_RANK(tensor); ++dimension) {
        elements *= dimensions[dimension];
      }
      Qnn_ClientBuffer_t clientBuffer = QNN_CLIENT_BUFFER_INIT;
      clientBuffer.dataSize = elements * elementSize(QNN_TENSOR_GET_DATA_TYPE(tensor));
      clientBuffer.data = std::calloc(1, clientBuffer.dataSize);
      if (clientBuffer.data == nullptr) return false;
      QNN_TENSOR_SET_CLIENT_BUF(tensor, clientBuffer);
    }
    return true;
  }

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
  std::vector<int32_t> forcedTokens{50264, 50359, 50363};
  int32_t decoderStartToken = 50258;
  int32_t endToken = 50257;
  int start = 1;
  int count = 0;
};

std::vector<int32_t> parseTokens(const std::string& value) {
  std::vector<int32_t> tokens;
  size_t start = 0;
  while (start < value.size()) {
    const size_t comma = value.find(',', start);
    tokens.push_back(std::stoi(value.substr(start, comma - start)));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  if (tokens.empty()) fail("forced token list cannot be empty");
  return tokens;
}

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
    else if (key == "--forced-tokens") options.forcedTokens = parseTokens(value);
    else if (key == "--decoder-start-token") options.decoderStartToken = std::stoi(value);
    else if (key == "--end-token") options.endToken = std::stoi(value);
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

int decoderLayerCount(const LoadedGraph& decoder) {
  int layers = 0;
  while (decoder.hasInput("k_cache_self_" + std::to_string(layers) + "_in")) ++layers;
  if (layers == 0) fail("could not detect decoder layers");
  return layers;
}

void initializeSelfCache(Qnn_Tensor_t& tensor) {
  const auto cache = buffer(tensor);
  switch (QNN_TENSOR_GET_DATA_TYPE(tensor)) {
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_FLOAT_32:
      std::memset(cache.data, 0, cache.dataSize);
      return;
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_UINT_8:
      std::memset(cache.data, kSelfCacheZeroPoint, cache.dataSize);
      return;
    default:
      fail("unsupported self-cache data type");
  }
}

void initializeAttentionMask(Qnn_Tensor_t& tensor) {
  const auto mask = buffer(tensor);
  switch (QNN_TENSOR_GET_DATA_TYPE(tensor)) {
    case QNN_DATATYPE_FLOAT_16:
      std::fill_n(static_cast<uint16_t*>(mask.data),
                  mask.dataSize / sizeof(uint16_t),
                  kFloat16NegativeHundred);
      return;
    case QNN_DATATYPE_FLOAT_32:
      std::fill_n(static_cast<float*>(mask.data), mask.dataSize / sizeof(float), -100.0f);
      return;
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_UINT_16:
      std::memset(mask.data, 0, mask.dataSize);
      return;
    default:
      fail("unsupported attention-mask data type");
  }
}

void revealAttentionPosition(Qnn_Tensor_t& tensor, size_t index) {
  const auto mask = buffer(tensor);
  switch (QNN_TENSOR_GET_DATA_TYPE(tensor)) {
    case QNN_DATATYPE_FLOAT_16:
      static_cast<uint16_t*>(mask.data)[index] = 0;
      return;
    case QNN_DATATYPE_FLOAT_32:
      static_cast<float*>(mask.data)[index] = 0.0f;
      return;
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_UINT_16:
      static_cast<uint16_t*>(mask.data)[index] = kMaskVisible;
      return;
    default:
      fail("unsupported attention-mask data type");
  }
}

float float16ToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
  const uint32_t exponent = (value >> 10) & 0x1F;
  uint32_t mantissa = value & 0x03FF;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int32_t floatExponent = 127 - 15 + 1;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        --floatExponent;
      }
      mantissa &= 0x03FF;
      bits = sign | (static_cast<uint32_t>(floatExponent) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7F800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

int32_t argmaxLogits(Qnn_Tensor_t& tensor) {
  const auto logitsBuffer = buffer(tensor);
  size_t bestIndex = 0;
  switch (QNN_TENSOR_GET_DATA_TYPE(tensor)) {
    case QNN_DATATYPE_FLOAT_16: {
      const auto* logits = static_cast<const uint16_t*>(logitsBuffer.data);
      const size_t count = logitsBuffer.dataSize / sizeof(uint16_t);
      float best = float16ToFloat(logits[0]);
      for (size_t i = 1; i < count; ++i) {
        const float value = float16ToFloat(logits[i]);
        if (value > best) {
          best = value;
          bestIndex = i;
        }
      }
      break;
    }
    case QNN_DATATYPE_FLOAT_32: {
      const auto* logits = static_cast<const float*>(logitsBuffer.data);
      const size_t count = logitsBuffer.dataSize / sizeof(float);
      bestIndex = std::max_element(logits, logits + count) - logits;
      break;
    }
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_UINT_16: {
      const auto* logits = static_cast<const uint16_t*>(logitsBuffer.data);
      const size_t count = logitsBuffer.dataSize / sizeof(uint16_t);
      bestIndex = std::max_element(logits, logits + count) - logits;
      break;
    }
    default:
      fail("unsupported logits data type");
  }
  return static_cast<int32_t>(bestIndex);
}

std::vector<int32_t> decode(LoadedGraph& decoder,
                            LoadedGraph& encoder,
                            const Options& options,
                            double& decoderMs) {
  const int decoderLayers = decoderLayerCount(decoder);
  for (int layer = 0; layer < decoderLayers; ++layer) {
    for (const char prefix : {'k', 'v'}) {
      const std::string cross = std::string(1, prefix) + "_cache_cross_" + std::to_string(layer);
      copyTensor(decoder.input(cross), encoder.output(cross), cross);
      const std::string self = std::string(1, prefix) + "_cache_self_" +
                               std::to_string(layer) + "_in";
      initializeSelfCache(decoder.input(self));
    }
  }

  auto tokenBuffer = buffer(decoder.input("input_ids"));
  auto positionBuffer = buffer(decoder.input("position_ids"));
  auto maskBuffer = buffer(decoder.input("attention_mask"));
  initializeAttentionMask(decoder.input("attention_mask"));
  auto* token = static_cast<int32_t*>(tokenBuffer.data);
  auto* position = static_cast<int32_t*>(positionBuffer.data);
  size_t maskElementBytes = 0;
  switch (QNN_TENSOR_GET_DATA_TYPE(decoder.input("attention_mask"))) {
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_UINT_16:
      maskElementBytes = sizeof(uint16_t);
      break;
    case QNN_DATATYPE_FLOAT_32:
      maskElementBytes = sizeof(float);
      break;
    default:
      fail("unsupported attention-mask data type");
  }
  const size_t maskElements = maskBuffer.dataSize / maskElementBytes;
  if (tokenBuffer.dataSize < sizeof(int32_t) || positionBuffer.dataSize < sizeof(int32_t) ||
      maskElements < static_cast<size_t>(kDecoderSteps + 1)) {
    fail("unexpected decoder scalar/mask tensor size");
  }

  std::vector<int32_t> tokens{options.decoderStartToken};
  decoderMs = 0.0;
  for (int step = 0; step < kDecoderSteps; ++step) {
    *token = tokens.back();
    *position = step;
    revealAttentionPosition(decoder.input("attention_mask"), kDecoderSteps - step);

    const auto started = Clock::now();
    decoder.execute();
    decoderMs += milliseconds(started, Clock::now());

    int32_t next = 0;
    if (step < static_cast<int>(options.forcedTokens.size())) {
      next = options.forcedTokens[step];
    } else {
      next = argmaxLogits(decoder.output("logits"));
    }
    tokens.push_back(next);
    if (next == options.endToken) break;

    for (int layer = 0; layer < decoderLayers; ++layer) {
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
      const std::vector<int32_t> tokens = decode(decoder, encoder, options, decoderMs);

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
