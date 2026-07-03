from datasets import load_dataset


class DatasetLoader:

    def __init__(
        self,
        dataset_name: str,
        config: str,
        streaming: bool = False
    ):
        self.dataset_name = dataset_name
        self.config = config
        self.streaming = streaming

    def load(self):

        return load_dataset(
            self.dataset_name,
            self.config,
            streaming=self.streaming
        )