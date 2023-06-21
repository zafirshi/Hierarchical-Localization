import os
from shutil import copyfile


def process_data(source: str, target: str) -> None:
    """
    Process original data_dir into query and db, which to be same with Aachen
    Args:
        source: original data_dir path string
        target: saved target_dir path string
    """
    all_dir = sorted(os.listdir(source))
    print(f"all_dir length {len(all_dir)}")

    for sub_dir_name in all_dir:
        sub_dir_path = os.path.join(source, sub_dir_name)
        all_files_name = sorted(os.listdir(sub_dir_path))
        for each_file in all_files_name:
            new_img_name = sub_dir_name + '_' + each_file
            old_img_path = os.path.join(sub_dir_path, each_file)
            if 'base' in sub_dir_name:
                new_img_path = os.path.join(target, f'db/{new_img_name}')
            elif 'test' in sub_dir_name:
                new_img_path = os.path.join(target, f'query/{new_img_name}')
            else:
                raise NameError('dir name should contain base or test')
            copyfile(old_img_path, new_img_path)
            print(f'{new_img_path} saved!!!')
        print(f'=============> {sub_dir_name} process over!!!')


if __name__ == '__main__':
    root = '/media/zafirshi/software/Datasets/images'
    save = '/media/zafirshi/software/Datasets/ChunXiRoad'
    process_data(root, save)
