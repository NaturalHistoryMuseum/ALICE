import luigi
from pathlib import Path
import time
import numpy as np
from datetime import datetime
import shutil

from alice.config import (
    INPUT_DIR, 
    NUM_CAMERA_VIEWS, 
    OUTPUT_DIR,
    LOG_DIR
)

from alice.tasks.base import BaseTask
from alice.tasks.image import ImageTask
from alice.models.labels import Specimen
from alice.utils.image import pad_image_to_width, save_image



    
class SpecimenTask(BaseTask):

    specimen_id = luigi.Parameter()
    
    def glob_image_paths(self):
        return [p.resolve() for p in Path(INPUT_DIR).glob(f'{self.specimen_id}*.*') if p.suffix.lower() in {".jpg", ".jpeg"}]    

    def requires(self):
        
        image_paths = self.glob_image_paths()        
        num_image_paths = len(image_paths)
        if num_image_paths != NUM_CAMERA_VIEWS:
            raise IOError(f'Only {num_image_paths} images found for {self.specimen_id}')
        
        for path in image_paths:
            yield ImageTask(path=path)

    def run(self):
        output = Path(self.output().path)
        output_dir = output.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        
        image_paths = [Path(i.path) for i in self.input()]    
        specimen = Specimen(self.specimen_id, paths=image_paths)
        results = specimen.process()
        
        with self.output().open('w') as log_file:
            
            log_file.write(f"Processing {self.specimen_id}\n")
            log_file.write(f"Start time {self._format_time(start_time)}\n")
            log_file.write(f"Processed {len(image_paths)} images\n")                 
        
            log_file.write(f"{len(results)} levels of labels detected\n")   
        
            for i, label in enumerate(results['labels']):
                save_image(label, output_dir / f'label-{i}.jpg')
        
            for i, result in results['composites'].items():
                labels = result.labels            
                # stacked_label = self._stack_labels(labels)
                # save_image(stacked_label, output_dir / f'labels-{i}.jpg')
                
                if result.composite is None:
                    log_file.write(f"Level {i}:No composite\n")
                else:
                    save_image(result.composite, output_dir / f'composite-{i}.jpg')
                    
                log_file.write(f"Level {i}: {len(labels)} usable labels\n")
                
            log_file.write("Execution time {:.3f} seconds\n".format(time.time() - start_time))   
            

    def output(self):     
        return luigi.LocalTarget(OUTPUT_DIR / self.specimen_id / f'{self.specimen_id}.log')
    
    @staticmethod
    def _format_time(timestamp):
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')    
    
    def _stack_labels(self, labels):
        # Gap between images
        margin= 20
        max_width = max([label.shape[1] for label in labels])
        padded_labels = [pad_image_to_width(label, max_width) for label in labels]
        padding = np.full((margin, max_width, 3), 255, dtype=np.uint8)

        interspersed = []
        for label in padded_labels:   
            interspersed.append(label)
            interspersed.append(padding)

        stacked_label = np.vstack(interspersed) 
        return stacked_label       
        
    
@SpecimenTask.event_handler(luigi.Event.FAILURE)
def on_specimen_task_error(task, exception):
    output_dir = Path(task.output().path).parent
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    error_log = output_dir.parent / f'{output_dir.stem}.error.log'
    
    with error_log.open('w') as f:
        f.write(f"Time: {datetime.now()}\n")
        f.write('Exception:')
        f.write(str(exception))
    
if __name__ == "__main__":
    luigi.build([SpecimenTask(specimen_id='014544308', force=True)], local_scheduler=True) 