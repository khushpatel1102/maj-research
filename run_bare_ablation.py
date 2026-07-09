s0 [direct: primary] factory_dev> db.utility_agents.find()
[
  {
    _id: ObjectId('691ef526b201cc7932e9496a'),
    agent_id: '691eed8766a08f7a747591c9',
    created_at: ISODate('2025-11-20T11:01:58.048Z'),
    updated_at: ISODate('2025-11-20T11:01:58.048Z'),
    agent_name: 'artifact_pptx_generator',
    description: 'Agent for generating PPT markdown content'
  },
  {
    _id: ObjectId('68e7c0f547bdf2b388e9496a'),
    agent_id: '68e7c040998896ef659a945a',
    created_at: ISODate('2025-10-09T14:04:37.625Z'),
    updated_at: ISODate('2025-10-09T14:04:37.625Z'),
    agent_name: 'artifact_pdf_generator',
    description: 'Agent for generating PDF markdown content'
  },
  {
    _id: ObjectId('69d78f020844d4a21a27eff0'),
    agent_id: '69d78f020844d4a21a27efef',
    agent_name: 'ase_simulation_evaluation',
    description: 'ASE: evaluates simulation traces against metrics',
    json_file: 'ase_simulation_evaluation.json',
    created_at: ISODate('2026-04-09T11:35:30.770Z'),
    updated_at: ISODate('2026-04-09T11:35:30.770Z')
  },
  {
    _id: ObjectId('69d78f020844d4a21a27eff2'),
    agent_id: '69d78f020844d4a21a27eff1',
    agent_name: 'ase_agent_hardening',
    description: 'ASE: proposes improved agent config from failures',
    json_file: 'ase_agent_hardening.json',
    created_at: ISODate('2026-04-09T11:35:30.790Z'),
    updated_at: ISODate('2026-04-09T11:35:30.790Z')
  },
  {
    _id: ObjectId('69b114dfe27b078603e401dd'),
    agent_id: '69b114dfe27b078603e401dc',
    agent_name: 'prompt_designer',
    description: 'Agent for designing AI prompts through conversation',
    json_file: 'generate_with_ai.json',
    created_at: ISODate('2026-03-11T07:08:15.261Z'),
    updated_at: ISODate('2026-03-11T07:08:15.261Z')
  },
  {
    _id: ObjectId('69f09d7bb759e2ca9b5f4a15'),
    agent_id: '69f09d7bb759e2ca9b5f4a14',
    agent_name: 'tool_builder',
    description: 'Agent for selecting and configuring tools for a given use-case',
    json_file: 'tool_builder_agent.json',
    created_at: ISODate('2026-04-28T11:43:55.927Z'),
    updated_at: ISODate('2026-04-28T11:43:55.927Z')
  },
  {
    _id: ObjectId('69d78f020844d4a21a27efee'),
    agent_id: '69d78f020844d4a21a27efed',
    agent_name: 'ase_simulation_generator',
    description: 'ASE: default test-case / simulation generator',
    json_file: 'ase_simulation_generator.json',
    created_at: ISODate('2026-04-09T11:35:30.752Z'),
    updated_at: ISODate('2026-04-09T11:35:30.752Z')
  },
  {
    _id: ObjectId('69b114dfe27b078603e401db'),
    agent_id: '69b114dfe27b078603e401da',
    agent_name: 'artifact_ppt_generator',
    description: 'Agent for generating PowerPoint presentations',
    json_file: 'pptx_agent.json',
    created_at: ISODate('2026-03-11T07:08:15.236Z'),
    updated_at: ISODate('2026-03-11T07:08:15.236Z')
  },
  {
    _id: ObjectId('69b114dfe27b078603e401e1'),
    agent_id: '69b114dfe27b078603e401e0',
    agent_name: 'openapi_schema_convertor',
    description: 'Agent for converting curl requests to OpenAPI schemas',
    json_file: 'openapi_convertor.json',
    created_at: ISODate('2026-03-11T07:08:15.301Z'),
    updated_at: ISODate('2026-03-11T07:08:15.301Z')
  },
  {
    _id: ObjectId('69f09d7cd1052dd1ec4868c9'),
    agent_id: '69f09d7cd1052dd1ec4868c8',
    agent_name: 'ase_simulation_generator_free_text',
    description: 'ASE: simulation generator returning free-text (non-JSON) test cases',
    json_file: 'ase_simulation_generator_free_text.json',
    created_at: ISODate('2026-04-28T11:43:56.067Z'),
    updated_at: ISODate('2026-04-28T11:43:56.067Z')
  },
  {
    _id: ObjectId('68e7c13447bdf2b388e9496c'),
    agent_id: '68e7c075aff617987f1861af',
    created_at: ISODate('2025-10-09T14:05:40.132Z'),
    updated_at: ISODate('2025-10-09T14:05:40.132Z'),
    agent_name: 'artifact_docx_generator',
    description: 'Agent for generating DOCX file'
  },
  {
    _id: ObjectId('68e7c11a47bdf2b388e9496b'),
    agent_id: '68e7c060998896ef659a9649',
    created_at: ISODate('2025-10-09T14:05:14.664Z'),
    updated_at: ISODate('2025-10-09T14:05:14.664Z'),
    agent_name: 'artifact_csv_generator',
    description: 'Agent for generating CSV file'
  },
  {
    _id: ObjectId('69f09d7bd1052dd1ec4868c5'),
    agent_id: '69f09d7bd1052dd1ec4868c4',
    agent_name: 'ase_persona_generator_free_text',
    description: 'ASE: persona generator accepting free-text agent descriptions',
    json_file: 'ase_persona_generator_free_text.json',
    created_at: ISODate('2026-04-28T11:43:55.991Z'),
    updated_at: ISODate('2026-04-28T11:43:55.991Z')
  },
  {
    _id: ObjectId('69f09d7cd1052dd1ec4868c7'),
    agent_id: '69f09d7cd1052dd1ec4868c6',
    agent_name: 'ase_scenario_generator_free_text',
    description: 'ASE: scenario generator accepting free-text agent descriptions',
    json_file: 'ase_scenario_generator_free_text.json',
    created_at: ISODate('2026-04-28T11:43:56.030Z'),
    updated_at: ISODate('2026-04-28T11:43:56.030Z')
  },
  {
    _id: ObjectId('69b114dfe27b078603e401df'),
    agent_id: '69b114dfe27b078603e401de',
    agent_name: 'magic_prompt',
    description: 'Agent for optimizing and rewriting prompts',
    json_file: 'magic_prompt.json',
    created_at: ISODate('2026-03-11T07:08:15.277Z'),
    updated_at: ISODate('2026-03-11T07:08:15.277Z')
  },
  {
    _id: ObjectId('69b114dfe27b078603e401e3'),
    agent_id: '69b114dfe27b078603e401e2',
    agent_name: 'json_schema_generator',
    description: 'Agent for generating valid JSON schemas',
    json_file: 'tool_schema_generator.json',
    created_at: ISODate('2026-03-11T07:08:15.327Z'),
    updated_at: ISODate('2026-03-11T07:08:15.327Z')
  },
  {
    _id: ObjectId('69b114dfe27b078603e401e5'),
    agent_id: '69b114dfe27b078603e401e4',
    agent_name: 'conversational_builder',
    description: 'Proxy agent for Anthropic Sonnet used in conversational builder',
    json_file: 'conversational_builder.json',
    created_at: ISODate('2026-03-11T07:08:15.347Z'),
    updated_at: ISODate('2026-03-11T07:08:15.347Z')
  },
  {
    _id: ObjectId('69d78f020844d4a21a27efea'),
    agent_id: '69d78f020844d4a21a27efe9',
    agent_name: 'ase_persona_generator',
    description: 'ASE: generates user personas for simulation',
    json_file: 'ase_persona_generator.json',
    created_at: ISODate('2026-04-09T11:35:30.715Z'),
    updated_at: ISODate('2026-04-09T11:35:30.715Z')
  },
  {
    _id: ObjectId('69d78f020844d4a21a27efec'),
    agent_id: '69d78f020844d4a21a27efeb',
    agent_name: 'ase_scenario_generator',
    description: 'ASE: generates scenarios for simulation',
    json_file: 'ase_scenario_generator.json',
    created_at: ISODate('2026-04-09T11:35:30.732Z'),
    updated_at: ISODate('2026-04-09T11:35:30.732Z')
  },
  {
    _id: ObjectId('69d78f020844d4a21a27eff4'),
    agent_id: '69d78f020844d4a21a27eff3',
    agent_name: 'ase_json_simulation_generator',
    description: 'ASE: simulation generator for json_schema response agents',
    json_file: 'ase_json_simulation_generator.json',
    created_at: ISODate('2026-04-09T11:35:30.811Z'),
    updated_at: ISODate('2026-04-09T11:35:30.811Z')
  }
]
Type "it" for more
rs0 [direct: primary] factory_dev>