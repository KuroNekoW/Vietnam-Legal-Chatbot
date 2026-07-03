from datasets import load_dataset


class DatasetLoader:
    """
    Responsible for loading datasets from Hugging Face.
    """

    def __init__(
        self,
        dataset_name: str,
        config_name: str,
        streaming: bool = False
    ):

        self.dataset_name = dataset_name
        self.config_name = config_name
        self.streaming = streaming

    def load(self):

        return load_dataset(
            self.dataset_name,
            self.config_name,
            streaming=self.streaming
        )