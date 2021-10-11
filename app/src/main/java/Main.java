import java.util.Random;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.data.market.MarketDataDescription;
import org.encog.ml.data.market.MarketDataType;
import org.encog.ml.data.market.MarketMLDataSet;
import org.encog.ml.data.market.TickerSymbol;
import org.encog.ml.data.market.loader.LoadedMarketData;
import org.encog.ml.data.market.loader.MarketLoader;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.ContainsFlat;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.propagation.scg.ScaledConjugateGradient;

public class Main {
	private static final Random PRNG = new Random();

	public static void main(String[] args) {
		double[] values = {0.998334166468282, 1.98669330795061, 2.9552020666134,
				3.89418342308651, 4.79425538604203, 5.64642473395035,
				6.44217687237691, 7.17356090899523, 7.83326909627483,
				8.41470984807897, 8.91207360061435, 9.32039085967226,
				9.63558185417193, 9.8544972998846, 9.97494986604055,
				9.99573603041505, 9.91664810452469, 9.73847630878195,
				9.46300087687414, 9.09297426825682, 8.63209366648874,
				8.0849640381959, 7.4570521217672, 6.75463180551151,
				5.98472144103956, 5.15501371821464, 4.2737988023383,
				3.34988150155905, 2.39249329213982, 1.41120008059867,
				0.415806624332905, -0.583741434275801, -1.57745694143248,
				-2.55541102026831, -3.5078322768962, -4.42520443294853,
				-5.29836140908493, -6.11857890942719, -6.87766159183974,
				-7.56802495307928,};

		double min = values[0];
		double max = values[0];
		for (double value : values) {
			if (value < min) {
				min = value;
			}
			if (value > max) {
				max = value;
			}
		}

		double[] range = {-0.5, 0.5};

		int inputSize = 10;
		int hiddenSize = 6;
		int outputSize = 3;

		double input[][] = new double[values.length
				- (inputSize + outputSize)][inputSize];
		double target[][] = new double[values.length
				- (inputSize + outputSize)][outputSize];
		for (int i = 0; i < values.length - (inputSize + outputSize); i++) {
			for (int j = 0; j < inputSize; j++) {
				input[i][j] = range[0] + (range[1] - range[0])
						* (values[i + j] - min) / (max - min);
			}
			for (int j = 0; j < outputSize; j++) {
				target[i][j] = range[0] + (range[1] - range[0])
						* (values[i + inputSize + j] - min) / (max - min);
			}
		}

		MLDataSet examples = new BasicMLDataSet(input, target);

		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, inputSize));
		network.addLayer(
				new BasicLayer(new ActivationTANH(), true, hiddenSize));
		network.addLayer(
				new BasicLayer(new ActivationTANH(), false, outputSize));
		network.getStructure().finalizeStructure();
		network.reset();

		Propagation[] propagations = {
				// TODO Cloning is not needed in real-time operational mode.
				new Backpropagation((ContainsFlat) network.clone(), examples),
				new ResilientPropagation((ContainsFlat) network.clone(),
						examples),
				new QuickPropagation((ContainsFlat) network.clone(), examples),
				new ScaledConjugateGradient((ContainsFlat) network.clone(),
						examples),
				new ManhattanPropagation((ContainsFlat) network.clone(),
						examples, PRNG.nextDouble())};

		System.out.println("Experiment start.");
		for (Propagation p : propagations) {
			System.out.println("" + p.getClass().getName());
			long start = System.currentTimeMillis();
			for (int c = 0; c < 10; c++) {
				while (System.currentTimeMillis() - start < 6000) {
					p.iteration();
				}
				System.out.println("" + p.getError());
			}
			p.finishTraining();
		}
		System.out.println("Experiment end.");
	}

}
