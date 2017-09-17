#!/bin/bash

java -cp commons-cli-1.2.jar:commons-math3-3.2.jar:. cs362.Learn -mode train -algorithm $1 -model_file $2 -data $3 -task $4
