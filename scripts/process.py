import luigi
from alice.tasks.process import ProcessTask

def main():
    task = ProcessTask()
    luigi.build([task])

if __name__ == "__main__":
    main()
    
