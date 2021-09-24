#!/bin/bash
for user in /Users/*.*;do
 if [ ! -f $user/Desktop/ROOT.txt ]; then
   cp /var/root/thing.txt $user/Desktop/ROOT.txt
   chmod a+r $user/Desktop/ROOT.txt
 fi
done

