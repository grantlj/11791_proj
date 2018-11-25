from boom.modules import Module
from multiprocessing import Pool
import os

def write_dict_to_json(obj,fn):
    import json
    pass
    if not fn.endswith(".json"):
        fn=fn+".json"

    json_str=json.dumps(obj,sort_keys=True,indent=2)
    with open(fn,"wb") as f:
        f.writelines(json_str)
    return

class qs_json_writer(Module):

    def __init__(self, module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs):
        super(qs_json_writer, self).__init__(module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs)

        self.content=None


    def process(self, job, data):

        #   get the question and context list, each item in the list is a dictionary
        self.content=data
        return data

    def save_job_data(self, job, data):

        path = job.output_base + '/' + self.output_file


        if not os.path.exists(job.output_base):
            os.mkdir(job.output_base)

        write_dict_to_json(self.content,path)
        return path