import datasets
from .task import register, Task

@register("proofwriter")
class proofwriter(Task):
    VERSION = 0
    DATASET_PATH = "json"
    DATASET_NAME = None

    # cache_dir = "./cache"
    
    train_files = './data/proofwriter/sematic_train.json'
    test_files = './data/proofwriter/sematic_test.json'

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
       
        testset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_files=self.test_files,
            cache_dir=cache_dir,
            download_mode=download_mode,
            split = "train" # zhuoyan: this dataset does not have train/val/test split, only the dataset object
        )

        trainset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_files=self.train_files,
            cache_dir=cache_dir,
            download_mode=download_mode,
            split = "train" # zhuoyan: this dataset does not have train/val/test split, only the dataset object
        )

        self.dataset = datasets.DatasetDict({
            "train": trainset,
            "validation": testset
        })

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        user =  "Given a set of rules and facts, you have to reason whether a statement is true or false. Here are some facts and rules: \n" + doc['facts'] + doc['rules'] + "Does it imply that the statement \"" + doc['question'] + "\" is True?\nAnswer with only True or False. The answer is: "
        return user


    def doc_to_target(self, doc):
        return str(doc["answer"])
    

    def fewshot_context(
        self, doc, num_fewshot, rnd=None, 
        description="You are a helpful assistant with abductive reasoning abilities. Given a set of rules and facts, you have to reason whether a statement is true or false.",
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        description = description + "\n\n" if description else ""
        
        if num_fewshot == 0:
            labeled_examples = ""
        else:
            fewshotex = self.fewshot_examples(k = num_fewshot, rnd = rnd)
            
            # print("fewshotex", fewshotex)
            labeled_examples = (
                "\n\n".join(
                    [
                        self.doc_to_text(doc) + self.doc_to_target(doc)
                        for doc in fewshotex
                    ]
                )
                + "\n\n"
            )

        # print("labeled_examples", [labeled_examples])

        example = self.doc_to_text(doc)

        # print("example", [example])


        # print("=====")
        return description + labeled_examples + example
    
    def construct_requests(self, doc, ctx):
        ll_positive, _ = rf.loglikelihood(ctx, " True")
        ll_negative, _ = rf.loglikelihood(ctx, " False")
        return ll_positive, ll_negative

    def process_results(self, doc, results):
        ll_positive, ll_negative = results
        pred = ll_positive > ll_negative
        gold = doc["answer"]
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}
