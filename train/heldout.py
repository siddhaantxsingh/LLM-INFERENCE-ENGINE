"""Write the held-out corpus (stdlib files never trained on) to a file."""
import sys

import data

_, held = data.split()
open(sys.argv[1], "wb").write(data.read(held))
print(len(held), "files")
