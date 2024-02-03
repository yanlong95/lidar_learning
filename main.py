import yaml

if __name__ == "__main__":
    config_file = './config/config.yml'
    config = yaml.safe_load(open(config_file))
    data_root = config['data_root']['data_root_folder']
    print(data_root)