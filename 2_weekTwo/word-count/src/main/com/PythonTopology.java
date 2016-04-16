/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import backtype.storm.Config;
import backtype.storm.LocalCluster;
import backtype.storm.StormSubmitter;
import backtype.storm.topology.TopologyBuilder;
import backtype.storm.tuple.Fields;
import bolts.SplitSentence;
import bolts.WordCount;
import spouts.ReadFileAsStreamSpout;

/**
 * This topology demonstrates Storm's stream groupings and multilang capabilities.
 */
public class PythonTopology {

        public static void main(String[] args) throws Exception {

        // topology setup
        TopologyBuilder builder = new TopologyBuilder();

        builder.setSpout("word-reader", new ReadFileAsStreamSpout());

        builder.setBolt("split", new SplitSentence(), 8).shuffleGrouping("word-reader");
        builder.setBolt("count", new WordCount(), 12).fieldsGrouping("split", new Fields("word"));

        // configuration
        Config conf = new Config();
        conf.put("wordsFile", args[0]);
        conf.put(Config.TOPOLOGY_DEBUG, false);
        conf.setDebug(false);

        // if only one arg, run in local mode
        if (args.length == 1){
            conf.setMaxTaskParallelism(3);

            LocalCluster cluster = new LocalCluster();
            cluster.submitTopology("word-count", conf, builder.createTopology());

            Thread.sleep(10000);

            cluster.shutdown();
        }
        // submit topology to storm
        else {
            conf.setNumWorkers(3);
            StormSubmitter.submitTopologyWithProgressBar(args[1], conf, builder.createTopology());
        }
    }
}
