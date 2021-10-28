# Author: Ernest Deak
# License: GPLv3
# This file is part of RT::Perceptron.
#
# RT::Perceptron is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RT::Perceptron is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

# along with RT::Perceptron.  If not, see <https://www.gnu.org/licenses/>.

package RT::Perceptron;
use strict;
use warnings FATAL => qw(all);
use Exporter qw(import);
use Carp qw(confess);

use Data::Dumper;

our @EXPORT_OK = qw(dot);

our $VERSION = "0.01";


sub dot($$){
  my ($matrix1, $matrix2) = @_;
  confess "Both parameters must be array references" unless ref($matrix1) eq "ARRAY" and ref($matrix2) eq "ARRAY";
  confess "Matrices must be of equal size" unless @$matrix1 == @$matrix2;
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
__END__
=pod

=head1 NAME

RT::Perceptron

=head1 SYNOPSIS

  use RT::Perceptron;
  use Test::More;

  my $rtp = RT::Perceptron->new(
    w => [0.0,0.0],   #weights
    b => 0.1,         #bias
    t => 0.0,         #threshold
    r => 0.04         #learning rate
  );
  # AND gate
  $rtp->trainw([
    1 => [1,1],
    0 => [0,0],
    0 => [0,1],
    0 => [1,0]
  ],1); # MAX 1 iteration

  # Check results
  is $rtp->input(0,0), 0;
  is $rtp->input(1,1), 1;
  is $rtp->input(0,1), 0;
  is $rtp->input(1,0), 0;

  done_testing;

=head1 COPYRIGHT AND LICENSE

Ernest Deak (C) 2021 - All rights reserved

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.