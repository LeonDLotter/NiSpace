from setuptools import setup
import versioneer
import glob

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    data_files=glob.glob("nispace/datalib/**")
)