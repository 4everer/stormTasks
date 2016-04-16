import backtype.storm.Config;
import backtype.storm.LocalCluster;
import backtype.storm.StormSubmitter;
import backtype.storm.generated.AlreadyAliveException;
import backtype.storm.generated.AuthorizationException;
import backtype.storm.generated.InvalidTopologyException;
import backtype.storm.topology.TopologyBuilder;
import backtype.storm.tuple.Fields;
import bolts.AggregateBolt;
import bolts.ReadFromUrlBolt;
import bolts.SklearnPredictBolt;
import bolts.XgbPredictBolt;
import spouts.SendUrlSpout;

/**
 * Created by g on 13/04/16.
 */
public class PageClassificationTopology {

    public static void main(String[] args) throws InterruptedException, InvalidTopologyException, AuthorizationException, AlreadyAliveException {

        // topology setup
        TopologyBuilder builder = new TopologyBuilder();

        builder.setSpout("send-url", new SendUrlSpout());

        builder.setBolt("get-content", new ReadFromUrlBolt(), 12).shuffleGrouping("send-url");
        builder.setBolt("vectoring-predict", new SklearnPredictBolt(), 8).shuffleGrouping("get-content");
        builder.setBolt("xgb-predict", new XgbPredictBolt(), 8).shuffleGrouping("get-content");
        builder.setBolt("aggregate", new AggregateBolt(), 4).fieldsGrouping(("vectoring-predict"), new Fields("url")).fieldsGrouping(("xgb-predict"), new Fields("url"));

        // configuration
        Config conf = new Config();
        conf.put("urlList", args[0]);
        conf.put(Config.TOPOLOGY_DEBUG, false);
        conf.setDebug(false);

        // if only one arg, run in local mode
        if (args.length == 1){
            conf.setMaxTaskParallelism(3);

            LocalCluster cluster = new LocalCluster();
            cluster.submitTopology("newsClassification", conf, builder.createTopology());

            Thread.sleep(40000);

            cluster.shutdown();
        }
        // submit topology to storm
        else {
            conf.setNumWorkers(3);
            StormSubmitter.submitTopologyWithProgressBar(args[1], conf, builder.createTopology());
        }
    }
}
