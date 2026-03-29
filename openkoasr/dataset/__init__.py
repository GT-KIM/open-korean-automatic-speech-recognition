from openkoasr.dataset.KsponSpeech.dataloader import KsponSpeechDataset

dataset_factory = {
    'KsponSpeech': KsponSpeechDataset,
}

class DatasetFactory:
    @staticmethod
    def load_dataset(config):
        dataset_class = dataset_factory.get(config.name, None)
        if not dataset_class:
            raise ValueError(f"Dataset {config.name} is not supported.")
        return dataset_class(config)
