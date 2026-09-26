#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char **out_error, NSError *error) {
    if (out_error == NULL) return;
    const char *text = error.localizedDescription.UTF8String ?: "unknown Core ML error";
    *out_error = strdup(text);
}

int coreml_run_smoke_model(const char *package_path,
                           float input,
                           float *output,
                           char **out_error) {
    @autoreleasepool {
        if (package_path == NULL || output == NULL) return 2;
        if (out_error != NULL) *out_error = NULL;

        NSString *path = [NSString stringWithUTF8String:package_path];
        NSURL *url = [NSURL fileURLWithPath:path];
        NSError *error = nil;
        NSURL *compiled_url = [MLModel compileModelAtURL:url error:&error];
        if (compiled_url == nil) {
            set_error(out_error, error);
            return 1;
        }

        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = MLComputeUnitsAll;
        MLModel *model = [MLModel modelWithContentsOfURL:compiled_url
                                           configuration:configuration
                                                   error:&error];
        if (model == nil) {
            set_error(out_error, error);
            return 1;
        }

        MLMultiArray *input_array = [[MLMultiArray alloc]
            initWithShape:@[@1]
                 dataType:MLMultiArrayDataTypeFloat32
                    error:&error];
        if (input_array == nil) {
            set_error(out_error, error);
            return 1;
        }
        input_array[0] = @(input);
        MLFeatureValue *input_value = [MLFeatureValue featureValueWithMultiArray:input_array];
        MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{@"x": input_value}
                         error:&error];
        if (provider == nil) {
            set_error(out_error, error);
            return 1;
        }

        id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider error:&error];
        if (prediction == nil) {
            set_error(out_error, error);
            return 1;
        }
        MLFeatureValue *result = [prediction featureValueForName:@"y"];
        if (result == nil || result.type != MLFeatureTypeMultiArray ||
            result.multiArrayValue.count != 1) {
            if (out_error != NULL) *out_error = strdup("expected scalar multi-array output named 'y'");
            return 1;
        }
        *output = result.multiArrayValue[0].floatValue;
        return 0;
    }
}

void coreml_free_string(char *value) {
    free(value);
}
