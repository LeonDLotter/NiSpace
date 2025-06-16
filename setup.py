from setuptools import setup
import versioneer
import glob

setup(
    cmdclass=versioneer.get_cmdclass(),
    data_files=glob.glob("nispace/datalib/**")
)