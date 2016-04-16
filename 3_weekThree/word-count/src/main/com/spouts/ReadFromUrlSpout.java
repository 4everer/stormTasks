package spouts;

import backtype.storm.spout.SpoutOutputCollector;
import backtype.storm.task.TopologyContext;
import backtype.storm.topology.IRichSpout;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Values;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;

import java.io.*;
import java.net.MalformedURLException;
import java.util.Map;
import java.util.Scanner;


public class ReadFromUrlSpout implements IRichSpout {

    private SpoutOutputCollector collector;
    private boolean completed = false;
    private TopologyContext context;
    private Scanner scanner;

    public void ack(Object msgId) {
        System.out.println("OK:"+msgId);
    }
    public void close() {}

    @Override
    public void activate() {

    }

    @Override
    public void deactivate() {

    }


    public void fail(Object msgId) {
        System.out.println("FAIL:"+msgId);
    }
    /**
     * The only thing that the methods will do It is emit each
     * file line
     */

    public void nextTuple() {
        /**
         * The nextuple it is called forever, so if we have been readed the file
         * we will wait and then return
         */
        if (completed) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                //Do nothing
            }
            return;
        }

        String str;

        // //Open the reader
        // BufferedReader reader = new BufferedReader(inputStreamReader);
        try{
            //Read all lines
            while(scanner.hasNextLine()){
                /**
                 * By each line emmit a new value with the line as a their
                 */
                // System.out.println(str);
                str = scanner.nextLine();
                this.collector.emit(new Values(str),str);
            }
        }catch(Exception e){
            throw new RuntimeException("Error reading tuple",e);
        }finally{
            completed = true;
        }
    }

    /**
     * We will create the file and get the collector object
     */
    public void open(Map conf, TopologyContext context,
                     SpoutOutputCollector collector) {
        try {
            String url = conf.get("url").toString();
            Document doc = Jsoup.connect(url).get();
            String text = doc.body().text();

            this.context = context;
            this.scanner = new Scanner(text);

            // URL url = new URL(conf.get("url").toString());
            // this.context = context;
            // this.inputStreamReader = new InputStreamReader(url.openStream());
        } catch (MalformedURLException e1) {
            e1.printStackTrace();
        } catch (IOException e1) {
            e1.printStackTrace();
        }
        this.collector = collector;
    }

    /**
     * Declare the output field "word"
     */
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("line"));
    }

    @Override
    public Map<String, Object> getComponentConfiguration() {
        return null;
    }

}

