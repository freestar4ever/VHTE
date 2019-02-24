#!/bin/bash

function get_random_file {
    echo "${1}/$(ls $1 | grep jpg | sort -R | tail -1)"
}

clear

echo "removing all known profiles..."
rm -Rf profile

echo "Start training for ==steve_jobs=="
python main.py --name steve_jobs --profile "faces/steve_jobs/40.jpg" "faces/steve_jobs/9.jpg" "faces/steve_jobs/10.jpg"

echo "Start training for ==Michelle Obama=="
python main.py --name michelle_obama --profile "faces/michelle_obama/18.jpg" "faces/michelle_obama/22.jpg" "faces/michelle_obama/36.jpg"

echo "Start training for ==Adrien Brody=="
python main.py --name adrien_brody --profile "faces/adrien_brody/13.jpg" "faces/adrien_brody/25.jpg" "faces/adrien_brody/47.jpg"

echo "--------------------"
echo "Done Training!"

steve_jobs="faces/steve_jobs/16.jpg"
michelle_obama="faces/michelle_obama/92.jpg"
adrien_brody="faces/adrien_brody/60.jpg"

echo "--------------------"
echo "This should be steve jobs: ${steve_jobs}"
python main.py ${steve_jobs}
echo "--------------------"
echo "This should be michelle obama: ${michelle_obama}"
python main.py ${michelle_obama}
echo "--------------------"
echo "This should should be adrien_brody: ${adrien_brody}"
python main.py ${adrien_brody}
echo "--------------------"

echo "End!"