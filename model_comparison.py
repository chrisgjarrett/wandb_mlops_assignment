import os
import wandb
import wandb.apis.reports as wbreport

# Get environment variables
try:
    ENTITY = os.environ["ENTITY"]
    PROJECT = os.environ["PROJECT"]
    RUN_ID = os.environ["RUN_ID"]
except:
    ENTITY = "chrisgjarrett"
    PROJECT = "mlops-course-assignment" 
    RUN_ID = "ihgq68ma"

# Path to model registry
path=os.path.join(ENTITY, 'model-registry/cancer-image-learner')

# Path to run with model to evaluate
run_path = os.path.join(ENTITY, PROJECT, RUN_ID)

# Connect to api and get run
api = wandb.Api()
run = api.run(run_path)

# Get the candidate model to be compared
candidate_model = [a for a in run.logged_artifacts() if a.type == 'model']

# New wandb session for generating report
wandb.init(entity=ENTITY, project=PROJECT)

# Get baseline model from registry.
api = wandb.Api()
versions = api.artifact_versions('model', name=path)
baseline_model = [v for v in versions if "baseline" in v.aliases]

report = wbreport.report.Report(
    entity=ENTITY,
    project = PROJECT,
    title='Compare runs',
    description = 'A demo of comparing runs programmatically'
)

pg = wbreport.PanelGrid(
    runsets=[
        wbreport.runset.Runset(ENTITY, PROJECT, "Run Comparisons").set_filters_with_python_expr("Name in ['misunderstood-night-71']")
    ],
    panels=[
        wbreport.RunComparer(diff_only='split', layout={'w': 24, 'h':15})
    ]
)

report.blocks = report.blocks[:1] + [pg] + report.blocks[1:]
report.save()
print(report.url)


# Create a report