import java.io.NotSerializableException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.propagation.scg.ScaledConjugateGradient;
import org.moeaframework.algorithm.AbstractAlgorithm;
import org.moeaframework.algorithm.PESA2;
import org.moeaframework.algorithm.PESA2.RegionBasedSelection;
import org.moeaframework.algorithm.single.AggregateObjectiveComparator;
import org.moeaframework.algorithm.single.DifferentialEvolution;
import org.moeaframework.algorithm.single.EvolutionStrategy;
import org.moeaframework.algorithm.single.GeneticAlgorithm;
import org.moeaframework.algorithm.single.LinearDominanceComparator;
import org.moeaframework.algorithm.single.SelfAdaptiveNormalVariation;
import org.moeaframework.core.Initialization;
import org.moeaframework.core.Problem;
import org.moeaframework.core.Selection;
import org.moeaframework.core.Solution;
import org.moeaframework.core.Variation;
import org.moeaframework.core.operator.GAVariation;
import org.moeaframework.core.operator.InjectedInitialization;
import org.moeaframework.core.operator.TournamentSelection;
//import org.moeaframework.core.operator.UniformCrossover;
import org.moeaframework.core.operator.UniformSelection;
import org.moeaframework.core.operator.binary.BitFlip;
import org.moeaframework.core.operator.binary.HUX;
import org.moeaframework.core.operator.permutation.Insertion;
import org.moeaframework.core.operator.real.DifferentialEvolutionSelection;
import org.moeaframework.core.operator.real.DifferentialEvolutionVariation;
import org.moeaframework.core.operator.real.UM;
import org.moeaframework.core.variable.EncodingUtils;
import org.moeaframework.core.variable.RealVariable;
import org.moeaframework.problem.AbstractProblem;
import io.jenetics.Mutator;
import io.jenetics.Optimize;
import io.jenetics.Genotype;
import io.jenetics.Phenotype;
import io.jenetics.DoubleGene;
import io.jenetics.MeanAlterer;
import io.jenetics.TournamentSelector;
import io.jenetics.RouletteWheelSelector;
import io.jenetics.EliteSelector;
import io.jenetics.DoubleChromosome;
//import io.jenetics.UniformCrossover;
import io.jenetics.engine.Limits;
import io.jenetics.engine.Codecs;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.engine.EvolutionStatistics;
import io.jenetics.util.DoubleRange;

public class Main {
	private static class MoeaNonlinearRegressionProblem
			extends
				AbstractProblem {

		public MoeaNonlinearRegressionProblem(int numberOfVariables,
				int numberOfObjectives, int numberOfConstraints) {
			super(numberOfVariables, numberOfObjectives, numberOfConstraints);
		}

		public MoeaNonlinearRegressionProblem(int numberOfVariables,
				int numberOfObjectives) {
			super(numberOfVariables, numberOfObjectives);
		}

		@Override
		public void evaluate(Solution solution) {
			double sum = 0;

			double[] coefficients = EncodingUtils.getReal(solution);

			for (int x = 0; x < values.length; x++) {
				double y = coefficients[0] * x + coefficients[1];

				for (int k = 2; k < coefficients.length; k += 3) {
					y += coefficients[k] * Math
							.sin(coefficients[k + 1] * x + coefficients[k + 2]);
				}

				sum += (y - values[x]) * (y - values[x]);
			}

			solution.setObjective(0, Math.sqrt(sum / values.length));
		}

		@Override
		public Solution newSolution() {
			Solution solution = new Solution(best.length, 1, 0);

			for (int i = 0; i < best.length; i++) {
				solution.setVariable(i,
						new RealVariable(best[i] + (PRNG.nextDouble() - 0.5D),
								MIN_RANGE, MAX_RANGE));
			}

			return solution;
		}
	}

	private static class JeneticsNonlinearRegressionProblem {
		public static double evaluate(double[] coefficients) {
			double sum = 0;

			for (int x = 0; x < values.length; x++) {
				double y = coefficients[0] * x + coefficients[1];

				for (int k = 2; k < coefficients.length; k += 3) {
					y += coefficients[k] * Math
							.sin(coefficients[k + 1] * x + coefficients[k + 2]);
				}

				sum += (y - values[x]) * (y - values[x]);
			}

			return Math.sqrt(sum / values.length);
		}
	}

	private static final Random PRNG = new Random();

	private static final long OPTIMIZATION_TIMEOUT = 60_000;

	private static final long PRINT_TIMEOUT = 1000;

	private static final double MIN_RANGE = -100_000;

	private static final double MAX_RANGE = +100_000;

	private static final int MIN_SIN_FUNCTIONS = 9;

	private static final int MAX_SIN_FUNCTIONS = 9;

	private static final int POPULATION_SIZE = 137;

	private static final double CROSSOVER_RATE = 0.95;

	private static final double MUTATION_RATE = 0.01;

	private static final double SCALING_FACTOR = 0.95;

	private static final double[] SIN = {0.998334166468282, 1.98669330795061,
			2.9552020666134, 3.89418342308651, 4.79425538604203,
			5.64642473395035, 6.44217687237691, 7.17356090899523,
			7.83326909627483, 8.41470984807897, 8.91207360061435,
			9.32039085967226, 9.63558185417193, 9.8544972998846,
			9.97494986604055, 9.99573603041505, 9.91664810452469,
			9.73847630878195, 9.46300087687414, 9.09297426825682,
			8.63209366648874, 8.0849640381959, 7.4570521217672,
			6.75463180551151, 5.98472144103956, 5.15501371821464,
			4.2737988023383, 3.34988150155905, 2.39249329213982,
			1.41120008059867, 0.415806624332905, -0.583741434275801,
			-1.57745694143248, -2.55541102026831, -3.5078322768962,
			-4.42520443294853, -5.29836140908493, -6.11857890942719,
			-6.87766159183974, -7.56802495307928,};

	private static final double[] BTC = {10242.330078, 10369.02832,
			10409.861328, 10452.399414, 10328.734375, 10677.754883,
			10797.761719, 10973.251953, 10951.820313, 10933.75293, 11095.870117,
			10934.925781, 10459.624023, 10539.457031, 10227.479492,
			10747.472656, 10702.237305, 10752.939453, 10771.641602,
			10712.462891, 10845.411133, 10785.010742, 10624.390625,
			10583.806641, 10567.919922, 10688.03418, 10799.77832, 10619.803711,
			10677.625, 11059.142578, 11296.082031, 11429.047852, 11426.602539,
			11502.828125, 11322.123047, 11355.982422, 11495.038086,
			11745.974609, 11913.077148, 12801.635742, 12971.548828,
			12931.574219, 13108.063477, 13031.201172, 13075.242188,
			13654.214844, 13271.298828, 13437.874023, 13546.532227,
			13780.995117, 13737.032227, 13550.451172, 13950.488281,
			14133.733398, 15579.729492, 15565.880859, 14833.753906,
			15479.595703, 15332.350586, 15290.90918, 15701.298828, 16276.44043,
			16317.808594, 16068.139648, 15955.577148, 16685.691406,
			17645.191406, 17803.861328, 17817.083984, 18621.316406,
			18642.232422, 18370.017578, 18365.015625, 19104.410156,
			18729.839844, 17153.914063, 17112.933594, 17719.634766,
			18178.322266, 19633.769531, 18801.744141, 19205.925781,
			19446.966797, 18698.384766, 19154.179688, 19343.128906,
			19191.529297, 18320.884766, 18553.298828, 18263.929688,
			18051.320313, 18806.765625, 19144.492188, 19246.919922,
			19418.818359, 21308.351563, 22806.796875, 23132.865234,
			23861.765625, 23474.455078, 22794.039063, 23781.974609,
			23240.203125, 23733.570313, 24677.015625, 26439.373047,
			26280.822266, 27081.810547, 27360.089844, 28841.574219,
			28994.009766, 29376.455078, 32129.408203, 32810.949219,
			31977.041016, 34013.613281, 36833.875, 39381.765625, 40788.640625,
			40254.21875, 38346.53125, 35516.359375, 33915.121094, 37325.109375,
			39156.707031, 36821.648438, 36163.648438, 35792.238281,
			36642.234375, 36050.113281, 35549.398438, 30817.625, 32985.757813,
			32064.376953, 32285.798828, 32358.613281, 32564.029297,
			30441.041016, 34318.671875, 34295.933594, 34270.878906,
			33114.578125, 33533.199219, 35510.820313, 37475.105469,
			36931.546875, 38138.386719, 39250.191406, 38886.828125,
			46184.992188, 46469.761719, 44898.710938, 47877.035156,
			47491.203125, 47114.507813, 48696.535156, 47944.457031,
			49207.277344, 52140.972656, 51675.980469, 55887.335938,
			56068.566406, 57532.738281, 54204.929688, 48835.085938,
			49709.082031, 47180.464844, 46344.773438, 46194.015625,
			45159.503906, 49612.105469, 48415.816406, 50522.304688, 48527.03125,
			48899.230469, 48918.679688, 51174.117188, 52272.96875, 54824.011719,
			55963.179688, 57821.21875, 57343.371094, 61221.132813, 59267.429688,
			55840.785156, 56825.828125, 58893.078125, 57850.441406,
			58332.261719, 58309.914063, 57517.890625, 54511.660156,
			54710.488281, 52726.746094, 51683.011719, 55137.566406,
			55974.941406, 55947.898438, 57750.132813, 58930.277344, 58926.5625,
			59098.878906, 59397.410156, 57604.839844, 58760.875, 59171.933594,
			58186.507813, 56099.914063, 58326.5625, 58253.777344, 59846.230469,
			60175.945313, 59890.019531, 63523.753906, 63075.195313,
			63258.503906, 61529.921875, 60701.886719, 56191.585938,
			55681.792969, 56471.128906, 53857.105469, 51739.808594,
			51143.226563, 50052.832031, 49077.792969, 54030.304688,
			55036.636719, 54858.089844, 53568.664063, 57714.664063,
			57825.863281, 56620.273438, 57214.179688, 53252.164063,
			57441.308594, 56413.953125, 57352.765625, 58877.390625,
			58250.871094, 55847.242188, 56714.53125, 49735.433594, 49682.980469,
			49855.496094, 46716.636719, 46415.898438, 43488.058594,
			42944.976563, 36753.667969, 40596.949219, 37371.03125, 37531.449219,
			34700.363281, 38795.78125, 38392.625, 39316.890625, 38507.082031,
			35684.15625, 34607.40625, 35658.59375, 37293.792969, 36699.921875,
			37599.410156, 39242.484375, 36880.15625, 35538.609375, 35835.265625,
			33589.519531, 33416.976563, 37389.515625, 36697.03125, 37340.144531,
			35555.789063, 39016.96875, 40427.167969, 40168.691406, 38341.421875,
			38099.476563, 35854.527344, 35563.140625, 35641.144531,
			31622.376953, 32515.714844, 33682.800781, 34659.105469,
			31594.664063, 32287.523438, 34679.121094, 34475.558594,
			35908.386719, 35035.984375, 33549.601563, 33854.421875,
			34665.566406, 35284.34375, 33723.507813, 34225.679688, 33889.605469,
			32861.671875, 33811.242188, 33509.078125, 34254.015625, 33125.46875,
			32723.845703, 32827.875, 31841.550781, 31397.308594, 31533.884766,
			31800.011719, 30838.285156, 29796.285156, 32138.873047,
			32305.958984, 33593.730469, 34290.292969, 35384.03125, 37276.035156,
			39503.1875, 39995.453125, 40027.484375, 42196.304688, 41460.84375,
			39907.261719, 39178.402344, 38213.332031, 39744.515625,
			40865.867188, 42832.796875, 44574.4375, 43791.925781, 46280.847656,
			45599.703125, 45576.878906, 44439.691406, 47810.6875, 47096.667969,
			47019.960938, 45936.457031, 44686.75, 44741.882813, 46723.121094,
			49327.074219, 48869.105469, 49291.675781, 49562.347656,
			47727.257813, 49002.640625, 46894.554688, 49072.585938, 48911.25,
			48834.851563, 47024.339844, 47099.773438, 48807.847656, 49288.25,
			50009.324219, 49937.859375, 51769.003906, 52660.480469,
			46827.761719, 45774.742188, 46476.492188,};

	private static double[] values = BTC;

	private static double[] best = {};

	private static void encog() {
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
				new Backpropagation((BasicNetwork) network.clone(), examples),
				new ResilientPropagation((BasicNetwork) network.clone(),
						examples),
				new QuickPropagation((BasicNetwork) network.clone(), examples),
				new ScaledConjugateGradient((BasicNetwork) network.clone(),
						examples),
				new ManhattanPropagation((BasicNetwork) network.clone(),
						examples, PRNG.nextDouble())};

		System.out.println("Experiment start.");
		for (Propagation p : propagations) {
			System.out.println("" + p.getClass().getName());
			long loop = 0;
			long second = 0;
			long time = System.currentTimeMillis();
			long start = System.currentTimeMillis();
			while (p.isTrainingDone() == false
					&& System.currentTimeMillis() - start < 1 * 60 * 1000) {
				loop++;
				p.iteration();
				if (System.currentTimeMillis() - time > 1000) {
					second++;
					System.out.println(
							second + "\t" + loop + "\t" + p.getError());
					time = System.currentTimeMillis();
				}
			}
			p.finishTraining();
		}
		System.out.println("Experiment end.");
	}

	private static void moea() {
		for (int numberOfSineFunctions = MIN_SIN_FUNCTIONS; numberOfSineFunctions < (MAX_SIN_FUNCTIONS
				+ 1); numberOfSineFunctions++) {
			double[] old = best;
			best = new double[2 + 3 * numberOfSineFunctions];
			for (int i = 0; i < old.length; i++) {
				best[i] = old[i];
			}
			for (int i = old.length; i < best.length; i++) {
				best[i] = PRNG.nextDouble() - 0.5D;
			}

			Problem problem = new MoeaNonlinearRegressionProblem(best.length,
					1);

			List<Solution> solutions = new ArrayList<>();
			for (int i = 0; i < POPULATION_SIZE; i++) {
				Solution solution = problem.newSolution();
				solutions.add(solution);
			}

			AggregateObjectiveComparator comparator = new LinearDominanceComparator();
			Initialization initialization = new InjectedInitialization(problem,
					POPULATION_SIZE, solutions);

			Selection[] selections = {new TournamentSelection(),
					// new UniformSelection(),
					// new DifferentialEvolutionSelection(),
			};

			Variation[] mutations = {new UM(MUTATION_RATE),
					// new BitFlip(mutationRate),
			};

			Variation[] crossovers = {
					new org.moeaframework.core.operator.UniformCrossover(
							CROSSOVER_RATE),
					// new HUX(crossoverRate),
			};

			AbstractAlgorithm[] algorithms = {

					// new EvolutionStrategy(problem, comparator,
					// initialization,
					// new SelfAdaptiveNormalVariation()),

					new GeneticAlgorithm(problem, comparator, initialization,
							selections[PRNG.nextInt(selections.length)],
							new GAVariation(
									crossovers[PRNG.nextInt(crossovers.length)],
									mutations[PRNG.nextInt(mutations.length)])),

					// new DifferentialEvolution(problem, comparator,
					// initialization,
					// new DifferentialEvolutionSelection(),
					// new DifferentialEvolutionVariation(crossoverRate,
					// scalingFactor)),

			};

			for (AbstractAlgorithm algorithm : algorithms) {
				System.out.println(algorithm.getClass().getName());

				long print = System.currentTimeMillis() - PRINT_TIMEOUT;
				long stop = System.currentTimeMillis() + OPTIMIZATION_TIMEOUT;
				while (System.currentTimeMillis() < stop) {
					algorithm.step();

					if (print > System.currentTimeMillis()) {
						continue;
					}

					System.out.print(System.currentTimeMillis());
					System.out.print("\t");
					System.out.print(algorithm.getNumberOfEvaluations());
					System.out.print("\t");
					System.out.print(numberOfSineFunctions);
					System.out.print("\t");
					System.out.print(
							algorithm.getResult().get(0).getObjective(0));
					System.out.print("\t");
					System.out.print(Arrays.toString(EncodingUtils
							.getReal(algorithm.getResult().get(0))));
					System.out.print("\n");

					print = System.currentTimeMillis() + PRINT_TIMEOUT;
				}

				best = EncodingUtils.getReal(algorithm.getResult().get(0));
			}
		}
	}

	private static void jenetics() {
		for (int numberOfSineFunctions = MIN_SIN_FUNCTIONS; numberOfSineFunctions < (MAX_SIN_FUNCTIONS
				+ 1); numberOfSineFunctions++) {
			double[] old = best;
			best = new double[2 + 3 * numberOfSineFunctions];
			for (int i = 0; i < old.length; i++) {
				best[i] = old[i];
			}
			for (int i = old.length; i < best.length; i++) {
				best[i] = PRNG.nextDouble() - 0.5D;
			}

			Engine<DoubleGene, Double> engine = Engine
					.builder(JeneticsNonlinearRegressionProblem::evaluate,
							Codecs.ofVector(
									DoubleRange.of(MIN_RANGE, MAX_RANGE),
									best.length))
					.populationSize(POPULATION_SIZE)
					.survivorsSelector(new TournamentSelector<>(5))
					.offspringSelector(new EliteSelector<>(3, new TournamentSelector<DoubleGene, Double>(3)))
					.optimize(Optimize.MINIMUM)
					.alterers(new Mutator<>(MUTATION_RATE),
							new io.jenetics.UniformCrossover<>(CROSSOVER_RATE))
					.build();

			EvolutionStatistics<Double, ?> statistics = EvolutionStatistics
					.ofNumber();

			long stop = System.currentTimeMillis() + OPTIMIZATION_TIMEOUT;
			for (long timeout = PRINT_TIMEOUT; timeout < OPTIMIZATION_TIMEOUT; timeout += PRINT_TIMEOUT) {
				Phenotype<DoubleGene, Double> found = engine.stream()
						.limit(Limits
								.byExecutionTime(Duration.ofMillis(timeout)))
						.peek(statistics)
						.collect(EvolutionResult.toBestPhenotype());
				
				System.out.print(System.currentTimeMillis());
				System.out.print("\t");
				System.out.print(statistics.evaluationDuration().count());
				System.out.print("\t");
				System.out.print(numberOfSineFunctions);
				System.out.print("\t");
				System.out.print(found.fitness());
				System.out.print("\t");
				System.out.print(found.genotype().chromosome());
				System.out.print("\n");

				// best = EncodingUtils.getReal(algorithm.getResult().get(0));
			}

		}
	}

	public static void main(String[] args) {
		// encog();
		// moea();
		jenetics();
	}
}
