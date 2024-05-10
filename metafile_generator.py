#Github link
url="https://github.com/manoja328/trojai-example"
# url="https://github.com/manoja328/trojai-rl-july"
#commit id

import subprocess
import re
process = subprocess.Popen(["git", "ls-remote", url], stdout=subprocess.PIPE)
stdout, stderr = process.communicate()
sha = re.split(r'\t+', stdout.decode('ascii'))[0]
print(sha)


template={}
template["nlayers_1"] =  {
    "description": "The hidden number of layers for training.",
    "type": "integer",
    "minimum": 1,
    "maximum": 3,
    "suggested_minimum": 2,
    "suggested_maximum": 3
}


meta_schema= {"$schema": "http://json-schema.org/draft-07/schema#",
              "title": "SRI Trinity Framework",
              "technique": "SRI NLP pipeline",
              "technique_description": "We are looking at the loss values of the model when triggers are proved",
              "technique_changes": "Some bug fixes on inferencing and loss calculation",
              "technique_type": ["Weight Analysis"],
              "commit_id": sha,
              "repo_name": url,
              "required": [],
              "additionalProperties": False,
              "type": "object",
              'properties': template}

params = {}
for k in template:
    params[k] = 2

import json
with open('metaparameters_schema.json','w') as f:
    json.dump(meta_schema,f)

with open('metaparameters.json','w') as f:
    json.dump(params,f)

print("Metaparameters generated and saved.....")
