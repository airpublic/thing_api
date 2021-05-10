from drf_yasg.generators import EndpointEnumerator
from drf_yasg.generators import OpenAPISchemaGenerator


class PublicEndpointEnumerator(EndpointEnumerator):
    def should_include_endpoint(self,
                                path,
                                callback,
                                app_name,
                                namespace,
                                url_name):

        print(path, callback, app_name, namespace, url_name)
        if url_name == 'v1-experiment-detail':
            return True
        else:
            #return super().should_include_endpoint(path, callback, app_name, namespace, url_name)
            return False


class PublicOpenAPISchemaGenerator(OpenAPISchemaGenerator):
    endpoint_enumerator_class = PublicEndpointEnumerator

    def determine_path_prefix(self, paths):
        return '/'
