# frozen_string_literal: true

# rules_engine.rb -- Ruby internal DSL for Chilean banking business rules
# (limits, blacklists, regulatory-inspired patterns).
#
# Two ways this file is used:
#   1. `require_relative 'rules_engine'` from RSpec (tests/ruby/rules_engine_spec.rb)
#      to call RuleSet#evaluate directly, in-process -- no JSON/IPC involved.
#   2. `ruby rules_engine.rb --server` from Python (src/python/bridge.py),
#      which keeps this process alive and evaluates one transaction per
#      line of newline-delimited JSON on STDIN/STDOUT. A persistent server
#      avoids paying Ruby's process-startup cost on every transaction --
#      the honest reason a one-shot `ruby rules_engine.rb` subprocess per
#      request would NOT be low latency (see README's architecture/latency
#      section for the measured difference).
require 'json'

# Illustrative thresholds inspired by Chilean AML/CMF norms (e.g. the
# cash-transaction reporting threshold under Ley 19.913 is commonly cited
# around UF 450). These are approximations for a synthetic-data demo, not
# verified legal figures -- see the README disclaimer. UF_TO_CLP is a fixed
# illustrative conversion, not a live indexed value.
UF_TO_CLP = 39_000.0
UF_REPORT_THRESHOLD_CLP = 450 * UF_TO_CLP

# Must stay outside the legit merchant pool (MER_00001..MER_00199) -- see
# the comment above BLACKLISTED_MERCHANT_IDS in generate_data.py for the
# real collision bug this was fixed from.
BLACKLISTED_MERCHANT_IDS = %w[MER_00666 MER_00777 MER_00999].freeze
HIGH_RISK_COUNTRY_CODES = %w[XX YY ZZ].freeze # placeholder codes, not real ISO countries

# A named, weighted business rule. `block` receives the transaction Hash
# (symbol keys) and returns truthy/falsy.
Rule = Struct.new(:name, :weight, :block) do
  def triggered?(transaction)
    !!block.call(transaction)
  end
end

# Registers rules via the `rule "name", weight: N do |txn| ... end` DSL and
# evaluates all of them against a transaction, producing a weighted verdict.
class RuleSet
  attr_reader :rules

  def initialize(flag_threshold: 50, &block)
    @rules = []
    @flag_threshold = flag_threshold
    instance_eval(&block) if block
  end

  def rule(name, weight:, &block)
    @rules << Rule.new(name, weight, block)
  end

  def evaluate(transaction)
    triggered = rules.select { |r| r.triggered?(transaction) }
    risk_score = [triggered.sum(&:weight), 100].min

    {
      risk_score: risk_score,
      flagged: risk_score >= @flag_threshold,
      triggered_rules: triggered.map(&:name),
    }
  end
end

RULES = RuleSet.new(flag_threshold: 50) do
  rule 'monto_excesivo', weight: 40 do |txn|
    txn[:amount_clp].to_f > 5_000_000
  end

  rule 'comercio_en_lista_negra', weight: 90 do |txn|
    BLACKLISTED_MERCHANT_IDS.include?(txn[:merchant_id])
  end

  rule 'pais_alto_riesgo', weight: 70 do |txn|
    HIGH_RISK_COUNTRY_CODES.include?(txn[:country_code])
  end

  rule 'estructuracion_subumbral', weight: 60 do |txn|
    amount = txn[:amount_clp].to_f
    amount > UF_REPORT_THRESHOLD_CLP * 0.85 &&
      amount < UF_REPORT_THRESHOLD_CLP &&
      txn[:txn_count_last_24h].to_i >= 3
  end

  rule 'rafaga_velocidad', weight: 50 do |txn|
    txn[:txn_count_last_1h].to_i >= 5
  end

  rule 'viaje_imposible', weight: 45 do |txn|
    !!txn[:is_impossible_travel]
  end
end

def run_server(input: $stdin, output: $stdout)
  output.sync = true
  input.each_line do |line|
    line = line.strip
    next if line.empty?

    begin
      transaction = JSON.parse(line, symbolize_names: true)
      verdict = RULES.evaluate(transaction)
      output.puts(verdict.to_json)
    rescue JSON::ParserError => e
      output.puts({ error: "invalid_json: #{e.message}" }.to_json)
    rescue StandardError => e
      output.puts({ error: "evaluation_error: #{e.message}" }.to_json)
    end
  end
end

if __FILE__ == $PROGRAM_NAME
  if ARGV.include?('--server')
    run_server
  else
    warn 'Usage: ruby rules_engine.rb --server  (reads newline-delimited JSON transactions from STDIN)'
    exit 1
  end
end
