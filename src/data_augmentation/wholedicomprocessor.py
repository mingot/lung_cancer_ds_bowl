import os


class WholeDicomProcessor(object):
    def __init__(self, input_dir, output_dir, dicom_name, TransformerClass, *args, **kwargs):
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._dicom_name = dicom_name
        self._TransformerClass = TransformerClass
        self._args = args
        self._kwargs = kwargs
    
    @property
    def input_dir(self):
        return self._input_dir
    
    @property
    def input_path(self):
        return os.path.join(self.input_dir, self.dicom_name)
    
    @property
    def output_dir(self):
        return self._output_dir
    
    @property
    def output_path(self):
        return os.path.join(self.output_dir, self.dicom_name)
    
    @property
    def dicom_name(self):
        return self._dicom_name
    
    @property
    def TransformerClass(self):
        return self._TransformerClass
    
    def check_paths(self):
        if not(os.path.exists(self.input_path)):
            raise Exception('Input path {} does not exist'.format(self.input_path))
        
        if os.path.exists(self.output_path):
            raise Exception('Output path {} already exists'.format(self.output_path))
        
        return True
    
    def run(self):
        self.check_paths()
        
        os.makedirs(self.output_path)
        filenames = os.listdir(self.input_path)
        input_files_paths = map(lambda elem: os.path.join(self.input_path, elem), filenames)
        output_files_paths = map(lambda elem: os.path.join(self.output_path, elem), filenames)
        
        for k, (input_file_path, output_file_path) in enumerate(zip(input_files_paths, output_files_paths)):
            print 'INFO: [{}/{}] Transforming {} to {}'.format(k + 1,
                                                                len(input_files_paths),
                                                                input_file_path,
                                                                output_file_path)
            transformer_instance = self.TransformerClass(input_path = input_file_path,
                                                            output_path = output_file_path,
                                                            *self._args,
                                                            **self._kwargs)
            transformer_instance.run()

