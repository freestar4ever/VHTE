#!/bin/bash

echo "removing all known profiles!"
rm -Rf profile

echo "Done! Start training..."
python main.py --profile "faces/steve_jobs/0.jpg" "faces/steve_jobs/9.jpg" "faces/steve_jobs/10.jpg"

who="faces/steve_jobs/16.jpg"
echo "Done! Let's see who is $who"
python main.py ${who}

echo "End!"

