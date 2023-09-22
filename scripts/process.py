import luigi
from alice.tasks.process import ProcessTask

def main():
    task = ProcessTask()
    luigi.build([task], local_scheduler=True)

if __name__ == "__main__":
    main()
    
