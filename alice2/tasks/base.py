import luigi

class BaseTask(luigi.Task):

    force = luigi.BoolParameter(default=False, significant=False)
            
    def complete(self):
        if self.force:
            return False
        
        return super().complete()  