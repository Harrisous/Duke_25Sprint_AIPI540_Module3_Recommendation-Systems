import pkg_resources

installed_packages = [pkg.key for pkg in pkg_resources.working_set]
print('numpy' in installed_packages)  # Should return True

import numpy
print(numpy.__file__)  # Prints the location of numpy module
