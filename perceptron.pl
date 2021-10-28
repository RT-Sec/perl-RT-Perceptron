# Standard perceptron example from AI::Perceptron CPAN package
# This file contains only example usage
use strict;
use warnings;

use AI::Perceptron;
use Test::More;

# AND gate
my $p = AI::Perceptron->new
              ->num_inputs(2)
              ->learning_rate(0.04)
              ->threshold(0.001)
              ->weights([0.5,0.5]);



$p->add_examples(
  [ -1 => -1, -1 ],
  [ -1 => -1,  1 ],
  [ -1 =>  1, -1 ],
  [  1 =>  1,  1 ]
);
$p->max_iterations(1)->train;

note "AND Gate test";
is $p->compute_output(1,1), 1;
is $p->compute_output(-1,1), -1;
is $p->compute_output(1,-1), -1;
is $p->compute_output(-1,-1), -1;

$p = AI::Perceptron->new
              ->num_inputs(2)
              ->learning_rate(0.01)
              ->threshold(0.4)
              ->weights([0.1,0.2]);



$p->add_examples(
  [ -1 => -1, -1 ],
  [  1 => -1,  1 ],
  [  1 =>  1, -1 ],
  [ -1 =>  1,  1 ]
);
$p->max_iterations(1000)->train;

note "XOR Gate";
is $p->compute_output(1,1), -1;
is $p->compute_output(-1,-1), -1;
is $p->compute_output(1,-1), 1;
is $p->compute_output(-1,1), 1;

$p = AI::Perceptron->new
              ->num_inputs(2)
              ->learning_rate(0.01)
              ->threshold(0.01)
              ->weights([0.001,0.002]);



$p->add_examples(
  [  -1 => -1, -1 ],
  [ 1 => -1,  1 ],
  [ 1 =>  1, -1 ],
  [  1 =>  1,  1 ]
);
$p->max_iterations(100)->train;

note "OR Gate";
is $p->compute_output(1,1), 1;
is $p->compute_output(-1,-1), -1;
is $p->compute_output(1,-1), 1;
is $p->compute_output(-1,1), 1;

$p = AI::Perceptron->new
              ->num_inputs(1)
              ->learning_rate(0.04)
              ->threshold(0.01)
              ->weights([0.5,0.5]);



$p->add_examples(
  [ -1 => 1 ],
  [ 1 => -1 ],
);
$p->max_iterations(5)->train;

note "NOT Gate";
is $p->compute_output(1), -1;
is $p->compute_output(-1), 1;

done_testing;