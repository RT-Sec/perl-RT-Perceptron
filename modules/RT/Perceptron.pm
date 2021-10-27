
package RT::Perceptron;
use strict;
use warnings FATAL => qw(all);
use Exporter qw(import);
use Carp;
use Data::Dumper;

our @EXPORT_OK = qw(dot);

our $VERSION = "0.01";

sub dot($$){
  my ($matrix1, $matrix2) = @_;
  carp "Both parameters must be array references" unless ref($matrix1) eq "ARRAY" and ref($matrix2) eq "ARRAY";
  carp "Matrices must be of equal size" unless @$matrix1 == @$matrix2;
  my $r = 0;
  for(0 .. $#$matrix1){
    $r += $$matrix1[$_] * $$matrix2[$_];
  }
  return $r;
}

sub new(){
  my ($pkg, %params) = @_;
  my $tron = bless {
    w => $params{w},
    b => $params{b},
    r => $params{r},
    t => $params{t}
  }, $pkg;
  $tron->{t} //= 0.0;
  return $tron;
}

sub compute($$) {
  my ($tron, $inputs) = @_;
  return dot($inputs, $tron->{w}) + $tron->{b};
}

sub activation($$) {
  my ($tron, $y) = @_;
  return $y >= $tron->{t} ? 1 : 0; # Step function
}

sub input(@){
  my ($tron, @params) = @_;
  $tron->activation($tron->compute(\@params));
}

sub trainw {
  my ($tron, $set, $max) = @_;
  $max //= -1;
  my $iter = 0;
  my @targets = grep { ref($_) ne "ARRAY" } @$set;
  my @inputs = grep { ref($_) eq "ARRAY" } @$set;
  my $inputs = \@inputs;
  #note explain $inputs;
  my $targets = \@targets;
  TRAIN:
  for my $wi(0 .. $#{$tron->{w}}){
    for my $xi(0 .. $#$targets){
      my $O = $tron->activation($tron->compute($inputs->[$xi]));
      for my $xj(0 .. $#{$inputs->[$xi]}){
        # print "Targets:";
        # print Dumper $targets;
        # print "inputs:";
        # print Dumper $inputs;
        # print "r:";
        # print Dumper $r;
        $tron->{w}->[$wi] += ($tron->{r} * ($targets->[$xi] - $O) * $inputs->[$xi]->[$xj]);
      }
    }
  }
  if($iter < $max){
    $iter++;
    goto TRAIN;
  }
  TRAINED:
}

1;