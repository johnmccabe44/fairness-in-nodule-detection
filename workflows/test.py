from metaflow import FlowSpec, step, IncludeFile, Parameter


class TestFlow(FlowSpec):

    

    @step
    def start(self):
        print('start')
        self.next(self.resample)

    @step
    def resample(self):

        self.samples = [1, 2, 3]
        self.next(self.calc_frocs, foreach='samples')

    @step
    def calc_frocs(self):
        print("Calculating frocs for sample %d" % self.input)
        self.sample = self.input
        self.frocs = [1, 2, 3]
        self.next(self.calc_froc, foreach='frocs')

    @step
    def calc_froc(self):
        self.froc = self.input
        self.sample = self.sample

        self.froc_score = self.froc * self.sample
        self.next(self.join)


    @step
    def join(self, inputs):
        
        self.froc_scores = [inp.froc_score for inp in inputs]
        self.next(self.join_samples)

    @step
    def join_samples(self, inputs):
        print("Joining samples", [inp.froc_scores for inp in inputs])
        self.next(self.end)
        
    @step
    def end(self):
        pass


if __name__ == '__main__':
    TestFlow()