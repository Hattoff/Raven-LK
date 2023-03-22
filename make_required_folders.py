import configparser
import os

config = configparser.ConfigParser()
config.read('config.ini')

def make_required_directories():
    for d in config['required_directories']:
        try:
            val = config['required_directories'][d]
            path = ".\%s" % val.replace("/", "\\") 
            if not os.path.exists(path):
                print('creating folder path: %s' % val)
            else:
                print('folder path exists: %s' % val)
            os.makedirs(path, exist_ok=True)
        except OSError as err:
            print(err)

make_required_directories()