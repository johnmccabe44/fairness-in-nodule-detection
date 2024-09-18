import random
from metaflow import FlowSpec, step, IncludeFile, Parameter


class TestFlow(FlowSpec):

    

    @step
    def start(self):
        print('start')
        self.next(self.define_sample)

    @step
    def define_sample(self):

        self.samples = [1, 2, 3]
        self.next(self.resample, foreach='samples')

    @step
    def resample(self):
        print("Calculating frocs for sample %d" % self.input)
        self.sample = self.input
        self.frocs = ['a', 'b', 'c']
        self.next(self.calc_froc, foreach='frocs')

    @step
    def calc_froc(self):
        self.froc = self.input
        self.sample = self.sample

        self.froc_score = random.random()
        self.next(self.join_froc)


    @step
    def join_froc(self, inputs):
        
        self.froc_scores = {str(inp.sample) + inp.froc : inp.froc_score for inp in inputs}
        self.next(self.join_samples)

    @step
    def join_samples(self, inputs):
        froc_scores = {}
        for inp in inputs:
            froc_scores.update(inp.froc_scores)
        print("Joining samples", froc_scores)
        self.next(self.end)
        
    @step
    def end(self):
        pass


if __name__ == '__main__':
    TestFlow()