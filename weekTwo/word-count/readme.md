readme
====

This is a demo for word count a txt file using storm

to run the demo, make sure you have python installed, then use `pip install jieba` to install `jieba`


direct into the word-count dir, first build the jar with Maven

`mvn package`

then

`$Storm_Home_Dir/bin/storm jar target/word-count-1.0-SNAPSHOT.jar PythonTopology ../testInput.txt`

The result will be in output.txt
