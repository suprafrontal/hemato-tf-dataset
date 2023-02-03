# import tomllib wait until python 3.11


def version():
    # wait until python 3.11
    # with open("pyproject.toml", "rb") as f:
    #     data = tomllib.load(f)
    #     print(data["tool.poetry"])
    with open("pyproject.toml", encoding="utf-8") as f:
        read_data = f.readline()
        read_data = f.readline()
        read_data = f.readline()
        return read_data.split(" ")[2][1:-2]


VERSION = version()


class Dataset:
    def version(self):
        return VERSION
