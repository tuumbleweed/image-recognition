import numpy as np
import matplotlib.pyplot as plt
import struct
import time

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def on_key_press(event):
    if event.key == 'left':
        print("Left arrow key pressed")
    elif event.key == 'right':
        print("Right arrow key pressed")


filename = 'tmp/t10k-images.idx3-ubyte'
images = read_idx(filename)

class ImageViewer:
    def __init__(self, images):
        self.images = images
        self.index = 0

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_image()
        plt.show()

    def update_image(self):
        self.ax.clear()
        self.ax.imshow(self.images[self.index], cmap='gray')
        self.ax.set_title(f'Image {self.index + 1}')
        self.fig.canvas.draw()

        print(self.images[self.index])

    def on_key(self, event):
        time.sleep(0.01)
        if event.key == 'right':
            self.index = (self.index + 1) % len(self.images)
        elif event.key == 'left':
            self.index = (self.index - 1) % len(self.images)
        self.update_image()

if __name__ == "__main__":
    viewer = ImageViewer(images)
