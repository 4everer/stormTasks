package bolts;

import backtype.storm.task.ShellBolt;
import backtype.storm.topology.IRichBolt;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.tuple.Fields;

import java.util.Map;


public class SplitSentence extends ShellBolt implements IRichBolt {

    public SplitSentence() {
            super("python", "splitsentence.py");
        }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
            declarer.declare(new Fields("word"));
        }

    @Override
    public Map<String, Object> getComponentConfiguration() {
            return null;
        }
}