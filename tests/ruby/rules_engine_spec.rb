# frozen_string_literal: true

require_relative '../../src/ruby/rules_engine'

RSpec.describe RuleSet do
  def base_transaction
    {
      amount_clp: 20_000,
      merchant_id: 'MER_00001',
      country_code: 'CL',
      txn_count_last_1h: 0,
      txn_count_last_24h: 1,
      is_impossible_travel: false,
    }
  end

  it 'does not flag an ordinary low-risk transaction' do
    verdict = RULES.evaluate(base_transaction)
    expect(verdict[:flagged]).to be(false)
    expect(verdict[:triggered_rules]).to be_empty
    expect(verdict[:risk_score]).to eq(0)
  end

  it 'flags an excessive amount' do
    txn = base_transaction.merge(amount_clp: 6_000_000)
    verdict = RULES.evaluate(txn)
    expect(verdict[:triggered_rules]).to include('monto_excesivo')
  end

  it 'flags a blacklisted merchant regardless of amount' do
    txn = base_transaction.merge(merchant_id: 'MER_00666', amount_clp: 1_000)
    verdict = RULES.evaluate(txn)
    expect(verdict[:flagged]).to be(true)
    expect(verdict[:triggered_rules]).to include('comercio_en_lista_negra')
    expect(verdict[:risk_score]).to eq(90)
  end

  it 'flags a high-risk country code' do
    txn = base_transaction.merge(country_code: 'XX')
    verdict = RULES.evaluate(txn)
    expect(verdict[:triggered_rules]).to include('pais_alto_riesgo')
  end

  it 'flags a sub-threshold structuring pattern' do
    # Just under the UF 450 reporting threshold, with several transactions
    # in the trailing 24h -- the classic "structuring" signature.
    txn = base_transaction.merge(amount_clp: UF_REPORT_THRESHOLD_CLP * 0.95, txn_count_last_24h: 4)
    verdict = RULES.evaluate(txn)
    expect(verdict[:triggered_rules]).to include('estructuracion_subumbral')
  end

  it 'does not flag structuring when the amount is comfortably below the threshold band' do
    txn = base_transaction.merge(amount_clp: UF_REPORT_THRESHOLD_CLP * 0.5, txn_count_last_24h: 4)
    verdict = RULES.evaluate(txn)
    expect(verdict[:triggered_rules]).not_to include('estructuracion_subumbral')
  end

  it 'flags a velocity burst' do
    txn = base_transaction.merge(txn_count_last_1h: 6)
    verdict = RULES.evaluate(txn)
    expect(verdict[:triggered_rules]).to include('rafaga_velocidad')
  end

  it 'flags impossible travel' do
    txn = base_transaction.merge(is_impossible_travel: true)
    verdict = RULES.evaluate(txn)
    expect(verdict[:triggered_rules]).to include('viaje_imposible')
  end

  it 'combines multiple triggered rules into a capped risk score' do
    txn = base_transaction.merge(
      amount_clp: 6_000_000,
      country_code: 'XX',
      txn_count_last_1h: 6,
      is_impossible_travel: true
    )
    verdict = RULES.evaluate(txn)
    expect(verdict[:risk_score]).to eq(100) # 40+70+50+45=205, capped at 100
    expect(verdict[:flagged]).to be(true)
  end

  it 'builds an independent RuleSet instance via the DSL without touching the global RULES' do
    custom = RuleSet.new(flag_threshold: 10) do
      rule 'siempre_dispara', weight: 15 do |_txn|
        true
      end
    end
    verdict = custom.evaluate(base_transaction)
    expect(verdict[:flagged]).to be(true)
    expect(verdict[:triggered_rules]).to eq(['siempre_dispara'])
  end
end
