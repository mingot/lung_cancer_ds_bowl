import dicom
import copy
import pylab


class _DicomTransformer(object):
    def __init__(self, input_path, output_path, unavailable_value = -2000):
        self.input_path = input_path
        self.output_path = output_path
        self._unavailable_value = unavailable_value


class DicomTransformer(_DicomTransformer):
    def __init__(self, *args, **kwargs):
        super(DicomTransformer, self).__init__(*args, **kwargs)
        
        self._dicom_object = dicom.read_file(self.input_path)
        self._input_pixels = copy.copy(self._dicom_object.pixel_array)
        self._input_pixels[self._input_pixels == self.unavailable_value] = 0
        self._output_pixels = None
    
    @property
    def unavailable_value(self):
        return self._unavailable_value
    
    @property
    def input_pixels(self):
        return self._input_pixels
    
    @property
    def output_pixels(self):
        if self._output_pixels == None:
            self._output_pixels = self.round(self.transform())
        
        return self._output_pixels
    
    @staticmethod
    def show_dicom(pixel_array, grid = False):
        if grid:
            raise NotImplementedError('show_dicom with grid option not implemented')
        else:
            array_to_show = pixel_array
        
        pylab.imshow(array_to_show, cmap = pylab.cm.bone)
        pylab.show()
    
    def round(self, pixel_array):
        return pixel_array.round().astype(self._input_pixels.dtype)
    
    def show_input(self):
        self.show_dicom(self.input_pixels)
    
    def show_output(self):
        self.show_dicom(self.output_pixels)
    
    def transform(self):
        raise NotImplementedError('transform method has not been implemented')
    
    def run(self):
        self._dicom_object.pixel_array = copy.copy(self.output_pixels)
        self._dicom_object.PixelData = self.output_pixels.tostring()
        self._dicom_object.save_as(self.output_path)
