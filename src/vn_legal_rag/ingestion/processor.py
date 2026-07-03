class DatasetProcessor:

    @staticmethod
    def inspect(dataset):

        print(dataset)

        first_split = list(dataset.keys())[0]

        sample = dataset[first_split][0]

        print()

        print("========== SAMPLE ==========")

        for key, value in sample.items():

            print(f"{key}: {type(value)}")