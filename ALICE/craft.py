from craft_text_detector import (
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
)

class Craft():

    def __init__(self, cuda=False):
        
        self.cuda = cuda
        # load models
        self.refine_net = load_refinenet_model(cuda=self.cuda)
        self.craft_net = load_craftnet_model(cuda=self.cuda)

    def detect(self, image):
        h, w = image.shape[:2]  
        # CRAFT internally scales to % 32 - so easier if we scale first
        # rather than trying to align image and heatmap
        h, w = self._calculate_32x_height_width(h, w)
        resized_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        # Set long size to max dim, otherwise CRAFT will scale
        long_size = max([h, w])

        prediction = get_prediction(
            image=resized_image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=self.cuda,
            long_size=long_size
        )

        heatmap = prediction["heatmaps"]['text_score_heatmap']
        # Heatmap will always be 2x smaller
        resized_heatmap = cv2.resize(heatmap, (resized_image.shape[1], resized_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        assert resized_heatmap.shape == resized_image.shape        
        prediction['image'] = resized_image
        prediction['resized_heatmap'] = resized_heatmap        
        return prediction

    @staticmethod
    def _scale_32x_dimension(d: int):
        # image height and width should be multiple of 32
        return round(d + (32 - (d % 32)))

    def _calculate_32x_height_width(self, h: int, w: int):
        shortest, longest = sorted([h, w])
        new_longest = self._scale_32x_dimension(longest)
        r = new_longest / longest
        new_shortest = self._scale_32x_dimension(r * shortest)
        return (new_longest, new_shortest) if h > w else (new_shortest, new_longest) 