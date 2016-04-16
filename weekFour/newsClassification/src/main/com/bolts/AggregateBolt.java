package bolts;

import backtype.storm.topology.BasicOutputCollector;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.base.BaseBasicBolt;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Tuple;
import backtype.storm.tuple.Values;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by g on 14/04/16.
 */

public class AggregateBolt extends BaseBasicBolt {

    Map<String, String> classfication = new HashMap<String, String>();

    @Override
    public void execute(Tuple input, BasicOutputCollector collector) {

        String title = input.getString(0);
        String url = input.getString(1);
        String p = input.getString(2);

        String predictions = classfication.get(title);

        if (predictions == null){
            predictions = "";
        }

        predictions = predictions.concat(" ## ").concat(p);

        classfication.put(title, predictions);
        collector.emit(new Values(url, predictions));

    }

    @Override
    public void cleanup() {
        System.out.println("===================");
        for (Map.Entry<String, String> entry : classfication.entrySet()) {
            System.out.println(entry.getKey() + " : " + entry.getValue() + "\n");
        }

        System.out.println("===================");

        BufferedWriter bw = null;

        try {
            // APPEND MODE SET HERE
            bw = new BufferedWriter(new FileWriter("../output.txt", true));
            for (Map.Entry<String, String> entry : classfication.entrySet()) {
                bw.write(entry.getKey() + " : " + entry.getValue() + "\n");
            }
            bw.newLine();
            bw.flush();
        } catch (IOException ioe) {
            ioe.printStackTrace();
        } finally {                       // always close the file
            if (bw != null) try {
                bw.close();
            } catch (IOException ioe2) {
                // just ignore it
            }
        }
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("title", "prediction"));
    }
}
