package bolts;

import backtype.storm.task.ShellBolt;
import backtype.storm.topology.IRichBolt;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.tuple.Fields;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;

/**
 * Created by g on 13/04/16.
 */

public class SklearnPredictBolt extends ShellBolt implements IRichBolt{

//    Integer id;
//    String name;
//
//    String title, url, prediction;

    public SklearnPredictBolt() {
        super("python", "vectorize_predict.py");
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("title", "url", "predictions"));
    }

    @Override
    public Map<String, Object> getComponentConfiguration() {
        return null;
    }

}
