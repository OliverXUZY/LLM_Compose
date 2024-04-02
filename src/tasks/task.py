

tasks_repo = {}
def register(name):
  def decorator(cls):
    tasks_repo[name] = cls
    return cls
  return decorator


class Task:
    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        self.download(data_dir, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None

    def fewshot_examples(self, k, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())
        return rnd.sample(self._training_docs, k)


    def fewshot_context(
        self, doc, num_fewshot, rnd=None, description=None
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
