package bolts;


import backtype.storm.topology.BasicOutputCollector;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.base.BaseBasicBolt;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Tuple;
import backtype.storm.tuple.Values;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;

import java.io.IOException;
import java.net.MalformedURLException;


public class ReadFromUrlBolt extends BaseBasicBolt {


    @Override
    public void execute(Tuple input, BasicOutputCollector collector) {
        String url = input.getString(0);

        try {
            Document doc = Jsoup.connect(url).get();
            String textContent = doc.body().text();
            String title = doc.title();
            collector.emit(new Values(title, url, textContent));

        } catch (MalformedURLException e1) {
            e1.printStackTrace();
        } catch (IOException e1) {
            e1.printStackTrace();
        }

    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("title", "url", "textContent"));
    }
}
