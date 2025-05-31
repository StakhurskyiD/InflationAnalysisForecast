# This is a sample Python script.
from src.EA.data.data_import_main import import_and_preprocess_research_data
from src.EA.data.features.train_prepare_and_fit import prepare_tree_data_for_xgb


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import_and_preprocess_research_data()
    prepare_tree_data_for_xgb()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
