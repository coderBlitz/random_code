#!/usr/bin/env python3
# We want, IBUUOCQ3Mh8zHFdoFDwEFldFDmsAUCsWVwdSK1MNF1dbCQ==

import sys
import base64


if __name__ == "__main__":

    k1 = "dawg_ctf"
    k2 = k1[::-1]

    if len(sys.argv) != 2:
        print("Invalid arguments.")
        sys.exit(1)

    if len(sys.argv[1]) < len(k1):
        print("Input too short.")
        sys.exit(1)

    in_data = bytearray(sys.argv[1].encode("ascii"))
    key = ""
    for i in range(len(k1)):
        if in_data[i] % 2 == 0:
            key += k1[i % len(k1)]
        else:
            key += k2[i % len(k2)]

    out_data = bytearray(len(in_data))
    for i, b in enumerate(in_data):
        out_data[i] = b ^ ord(key[i % len(key)])

    out_data = base64.b64encode(bytes(out_data))
    print(out_data.decode("ascii"))
