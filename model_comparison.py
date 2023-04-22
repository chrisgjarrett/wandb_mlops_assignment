import os
import wandb
import wandb.apis.reports as wbreport

# PROJECT="mlops-cicd-course"

try:
    ENTITY = os.environ["ENTITY"]
    PROJECT = os.environ["PROJECT"]
    RUN_ID = os.environ["RUN_ID"]
except:
    ENTITY = "chrisgjarrett"
    PROJECT = "mlops-course-assignment" 
    RUN_ID = "ihgq68ma"

path = os.path.join(ENTITY, PROJECT, RUN_ID)
# "chrisgjarrett/mlops-course-assignment/ihgq68ma"

api = wandb.Api()
run = api.run(path)

artefact = [a for a in run.logged_artifacts() if a.type == 'model']

wandb.init(entity=ENTITY, project=PROJECT)

# Get model from registry. TODO: Run id should come from comment
# versions = api.artifact_versions('model', path)

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