import torch

class ToyDataset(object):
    """
    A dataset class for generating toy images with lines.

    Args:
        transform (str): Name of the transformation to be applied to the images.
        size (int): Number of samples in the dataset.
        img_shape (tuple): Shape of the generated images.
        cell_size (int): Size of each cell in the grid.
        line_length (int): Length of the lines.
        prob_for_line (float): Probability of a cell having a line.
        train (bool): Flag indicating whether the dataset is for training or not.
    """

    def __init__(self,
        transform: str = '',
        size: int = 1024,
        img_shape: tuple = (48, 48),
        cell_size: int = 8,
        line_length: int = 5,
        prob_for_line: float = 0.5,
        train: bool = True):
        super().__init__()

        self.transform_name = transform
        self.size = size
        self.img_shape = img_shape
        self.cell_size = cell_size
        self.line_length = line_length
        self.prob_for_line = prob_for_line
        if train:
            seed_offset = 9982634964
        else:
            seed_offset = 1408368308
        self.seed_offset = seed_offset

        self.n_classes = 2
        self.class_names = ["vertical", "horizontal"]
        
        
    def prepare(self, shared_modules):
        """
        Prepares the dataset by setting the transformation.
        """
        self.transform = None
        if self.transform_name:
            self.transform = shared_modules[self.transform_name]


    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.
        """
        generator = torch.random.manual_seed(idx + self.seed_offset)
        out_image, label = self.generate_image(idx, generator)
        if self.transform is not None:
            out_image = self.transform(out_image)
        sample = {'data' : out_image, 'label' : label, 'id' : idx}
        return sample


    def generate_image(self, idx, generator):
        """
        Generates an image with lines.
        """
        img = torch.zeros(self.img_shape)
        hor_count = 0
        ver_count = 0
        for ix in range(1, int(self.img_shape[1]/self.cell_size) - 1):
            for iy in range(1, int(self.img_shape[0]/self.cell_size) - 1):
                if torch.rand(1, generator=generator) < self.prob_for_line:
                    horizontal = torch.rand(1, generator=generator) < 0.5
                    if horizontal:
                        hor_count +=1
                    else:
                        ver_count +=1
                    x_factor = 0.1 if horizontal else 0.8
                    y_factor = 0.5 if horizontal else 0.2
                    pos_x = int((ix + x_factor * torch.rand(1, generator=generator)) * self.cell_size)
                    pos_y = int((iy + y_factor * torch.rand(1, generator=generator)) * self.cell_size)
                    self.draw_line(img, pos_x, pos_y, horizontal)
        label = 1 if hor_count >= ver_count else 0
        label = torch.tensor(label, dtype=torch.long)
        if hor_count == ver_count:
            return self.generate_image(idx, generator)
        return img.unsqueeze(0), label


    def draw_line(self, img, ix, iy, horizontal):
        """
        Draws a line on the image.
        """
        if horizontal:
            img[iy,ix:ix+self.line_length] = 1.0
        else:
            img[iy:iy+self.line_length, ix] = 1.0

